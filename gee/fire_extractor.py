"""
Fire Risk Data Extractor using Google Earth Engine

Purpose: Extract fire risk features for ML classification model
Region: Uttarakhand, India (forest areas only)
Period: March-June 2023 & 2024 (fire seasons)
Output: CSV with fire occurrence labels + environmental features

EECU Budget: ~25-30 EECU (well under 150 limit)
Export Time: 15-25 minutes
Expected Rows: 150k-250k

Data Sources:
    1. MODIS Active Fire (MOD14A1) - Fire labels
    2. MODIS LST (MOD11A1) - Thermal stress
    3. MODIS Vegetation (MOD13Q1) - Fuel availability
    4. ERA5-Land Weather - Meteorological drivers
    5. SRTM DEM - Terrain controls
    6. MODIS Land Cover (MCD12Q1) - Forest mask
    7. WorldPop Population - Human influence (optional)
    8. FLDAS Soil Moisture - Dryness indicator (optional)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

import ee
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



@dataclass
class FireDataConfig:
    """Configuration for fire data extraction."""
    
    # Region settings
    region_name: str = 'Uttarakhand'
    country: str = 'India'
    
    # Google Cloud Project (loaded from .env)
    gee_project: str = field(default_factory=lambda: os.getenv('GEE_PROJECT', 'earthengine-legacy'))
    
    # Date ranges (fire season: March-June)
    date_ranges: list[tuple[str, str]] = field(default_factory=lambda: [
        ('2023-03-01', '2023-06-30'),
        ('2024-03-01', '2024-06-30')
    ])
    
    # Sampling parameters
    sample_scale: int = 1000  # 1km resolution (MODIS native)
    samples_per_week_fire: int = 8000  # Fire pixels (rare, sample more)
    samples_per_week_nofire: int = 4000  # Non-fire pixels
    
    # Export settings
    export_folder: str = 'HackEarth_Fire'
    export_description: str = 'uttarakhand_fire_data'
    
    # Optional features (set False to save EECU)
    include_population: bool = True
    include_soil_moisture: bool = False
    
    # Fire detection threshold
    fire_confidence_threshold: int = 7  # FireMask > 7 = high confidence




class GEEDatasets:
    """Google Earth Engine dataset collection IDs and bands."""
    
    # Region boundary
    ADMIN_BOUNDARIES = 'FAO/GAUL/2015/level1'
    
    # Static datasets
    SRTM_DEM = 'USGS/SRTMGL1_003'
    LAND_COVER = 'MODIS/061/MCD12Q1'
    POPULATION = 'WorldPop/GP/100m/pop'
    
    # Historical Burned Area (for features, NOT labels)
    BURNED_AREA = {
        'collection': 'MODIS/061/MCD64A1',
        'bands': ['BurnDate'],
        'description': 'MODIS Burned Area (500m, monthly)',
        'historical_years': (2018, 2022)  # Strictly before 2023-2024 labels
    }
    
    # Dynamic datasets
    FIRE_THERMAL = {
        'collection': 'MODIS/061/MOD14A1',
        'bands': ['FireMask'],
        'description': 'MODIS Thermal Anomalies (1km, 8 day composite)'
    }
    
    LST = {
        'collection': 'MODIS/061/MOD11A1',
        'bands': ['LST_Day_1km', 'LST_Night_1km'],
        'description': 'Land Surface Temperature (1km, daily)'
    }
    
    VEGETATION = {
        'collection': 'MODIS/061/MOD13Q1',
        'bands': ['NDVI', 'EVI'],
        'description': 'Vegetation Index (250m, 16-day)'
    }
    
    WEATHER = {
        'collection': 'ECMWF/ERA5_LAND/DAILY_AGGR',
        'bands': [
            'temperature_2m_max',
            'total_precipitation_sum',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m'
        ],
        'description': 'ERA5-Land Weather (11km, daily)'
    }
    
    SOIL_MOISTURE = {
        'collection': 'NASA/FLDAS/NOAH01/C/GL/M/V001',
        'bands': ['SoilMoi00_10cm_tavg'],
        'description': 'Soil Moisture (10km, monthly)'
    }
    
    # Export column schema
    EXPORT_PROPERTIES = [
        # Target variable
        'fire_occurred',
        
        # Historical burn features (2018-2022, strictly before labels)
        'burned_last_5yr', 'burn_count_5yr', 'years_since_last_burn',
        
        # Vegetation features
        'NDVI', 'EVI', 'ndvi_30day_mean',
        
        # Temperature features
        'LST_Day_1km', 'LST_Night_1km', 'lst_day_max', 'temp_max',
        
        # Precipitation features
        'precip_7day', 'precip_30day',
        
        # Wind
        'wind_speed',
        
        # Terrain
        'slope', 'aspect', 'elevation',
        
        # Optional
        'population', 'soil_moisture', 'land_cover',
        
        # Metadata
        'week_start', 'week_end',
        
        # Coordinates
        '.geo'
    ]




class FireDataExtractor:
    """
    Main class for extracting fire risk data from Google Earth Engine.
    
    Implements EECU-optimized extraction with:
    - Early filtering (bounds, dates, bands)
    - Server-side operations (avoid .getInfo() in loops)
    - Stratified sampling (more fire, less non-fire)
    - Batch export (cheaper than online operations)
    
    Usage:
        config = FireDataConfig()
        extractor = FireDataExtractor(config)
        task = extractor.run()
        # Monitor task.status() for progress
    """
    
    def __init__(self, config: Optional[FireDataConfig] = None):
        """
        Initialize the fire data extractor.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or FireDataConfig()
        self._region: Optional[ee.Geometry] = None
        self._static_data: dict = {}
        self._forest_mask: Optional[ee.Image] = None
        
        # Initialize Earth Engine
        self._initialize_ee()
    
    def _initialize_ee(self) -> None:
        """Initialize Google Earth Engine with error handling."""
        try:
            ee.Initialize(project=self.config.gee_project)
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Earth Engine initialization failed: {e}")
            logger.info("Run: ee.Authenticate() first if not authenticated")
            raise
    
    def _load_region(self) -> ee.Geometry:
        """
        Load the region of interest boundary.
        
        Returns:
            ee.Geometry: Region boundary geometry
        """
        if self._region is not None:
            return self._region
        
        logger.info(f"Loading {self.config.region_name} boundary...")
        
        region = ee.FeatureCollection(GEEDatasets.ADMIN_BOUNDARIES) \
            .filter(ee.Filter.eq('ADM1_NAME', self.config.region_name))
        
        self._region = region.geometry()
        logger.info(f"Region loaded: {self.config.region_name}")
        
        return self._region
    
    def _load_static_datasets(self) -> dict:
        """
        Load static (time-invariant) datasets.
        
        Optimizations:
        - Clipped immediately to reduce data volume
        - Bands selected upfront
        
        Returns:
            dict: Dictionary of static ee.Image objects
        """
        if self._static_data:
            return self._static_data
        
        region = self._load_region()
        logger.info("Loading static datasets...")
        
        # 1. Terrain (SRTM 30m DEM)
        dem = ee.Image(GEEDatasets.SRTM_DEM).clip(region)
        self._static_data['elevation'] = dem.select('elevation')
        self._static_data['slope'] = ee.Terrain.slope(dem).rename('slope')
        self._static_data['aspect'] = ee.Terrain.aspect(dem).rename('aspect')
        logger.info("  - Terrain data loaded (DEM, slope, aspect)")
        
        # 2. Land Cover (Forest mask)
        landcover = ee.ImageCollection(GEEDatasets.LAND_COVER) \
            .sort('system:time_start', False) \
            .first() \
            .select('LC_Type1') \
            .clip(region)
        
        self._static_data['land_cover'] = landcover.rename('land_cover')
        
        # IGBP Classification: 1-5 are forest types, 6 is closed shrublands
        # 1=Evergreen Needleleaf, 2=Evergreen Broadleaf, 
        # 3=Deciduous Needleleaf, 4=Deciduous Broadleaf, 5=Mixed Forest
        # 6=Closed Shrublands (many Uttarakhand fires occur here)
        self._forest_mask = landcover.lte(6).selfMask()
        logger.info("  - Land cover loaded (forest mask: LC_Type1 <= 6)")
        
        # 3. Population (Optional)
        if self.config.include_population:
            population = ee.ImageCollection(GEEDatasets.POPULATION) \
                .filterBounds(region) \
                .mosaic() \
                .clip(region) \
                .rename('population')
            self._static_data['population'] = population
            logger.info("  - Population data loaded")
        else:
            self._static_data['population'] = ee.Image.constant(0).rename('population')
            logger.info("  - Population data skipped (disabled)")
        
        # 4. Historical Burned Area Features (2018-2022, strictly before 2023-2024 labels)
        # These are FEATURES, not labels - used to capture historical fire susceptibility
        burned_area_features = self._load_historical_burned_area(region)
        self._static_data.update(burned_area_features)
        logger.info("  - Historical burned area features loaded (2018-2022)")
        
        return self._static_data
    
    def _load_historical_burned_area(self, region: ee.Geometry) -> dict:
        """
        Load historical burned area features from MODIS MCD64A1 (2018-2022).
        
        Creates three features per pixel:
        - burned_last_5yr: Binary (1 if burned at least once, else 0)
        - burn_count_5yr: Number of times burned (0-5)
        - years_since_last_burn: Years since most recent burn (capped at 5)
        
        Uses data STRICTLY from 2018-2022 to avoid label leakage with 2023-2024.
        
        Args:
            region: Region geometry to clip to
            
        Returns:
            dict: Dictionary with burned area ee.Image features
        """
        start_year, end_year = GEEDatasets.BURNED_AREA['historical_years']
        reference_year = 2023  # First label year
        
        # Load burned area collection for historical period
        burned_collection = ee.ImageCollection(GEEDatasets.BURNED_AREA['collection']) \
            .filterDate(f'{start_year}-01-01', f'{end_year}-12-31') \
            .filterBounds(region) \
            .select('BurnDate')
        
        # Process each year to get binary burned/not burned
        yearly_burns = []
        for year in range(start_year, end_year + 1):
            year_start = f'{year}-01-01'
            year_end = f'{year}-12-31'
            
            # Get max BurnDate for the year (>0 means burned)
            year_burned = burned_collection \
                .filterDate(year_start, year_end) \
                .max() \
                .gt(0) \
                .unmask(0) \
                .rename(f'burned_{year}')
            
            yearly_burns.append({
                'year': year,
                'image': year_burned
            })
        
        # 1. burned_last_5yr: Binary - burned at least once in 2018-2022
        burn_any = yearly_burns[0]['image']
        for yb in yearly_burns[1:]:
            burn_any = burn_any.Or(yb['image'])
        burned_last_5yr = burn_any.rename('burned_last_5yr').clip(region)
        
        # 2. burn_count_5yr: Count of years burned (0-5)
        burn_sum = yearly_burns[0]['image'].toInt()
        for yb in yearly_burns[1:]:
            burn_sum = burn_sum.add(yb['image'].toInt())
        burn_count_5yr = burn_sum.rename('burn_count_5yr').clip(region)
        
        # 3. years_since_last_burn: Years since most recent burn (cap at 5 if never)
        # Find the most recent year with a burn
        # Start with a default of reference_year - start_year (max years = 5)
        max_years = reference_year - start_year
        years_since = ee.Image.constant(max_years).toFloat()
        
        # Update years_since for each year where burn occurred
        for yb in yearly_burns:
            year = yb['year']
            years_val = reference_year - year
            # Where burned this year, set years_since to this value (if smaller)
            years_since = years_since.where(
                yb['image'].eq(1),
                ee.Image.constant(years_val).toFloat()
            )
        
        years_since_last_burn = years_since.rename('years_since_last_burn').clip(region)
        
        return {
            'burned_last_5yr': burned_last_5yr,
            'burn_count_5yr': burn_count_5yr,
            'years_since_last_burn': years_since_last_burn
        }
    
    def _load_dynamic_collections(
        self, 
        start: ee.Date, 
        end: ee.Date
    ) -> dict[str, ee.ImageCollection]:
        """
        Load dynamic (time-series) dataset collections.
        
        Optimizations:
        - Filtered by bounds immediately
        - Filtered by date (with 30-day lookback)
        - Only required bands selected
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            dict: Dictionary of filtered ee.ImageCollection objects
        """
        region = self._load_region()
        lookback_start = start.advance(-30, 'day')
        
        collections = {}
        
        # Fire Thermal Anomalies
        collections['fires'] = ee.ImageCollection(GEEDatasets.FIRE_THERMAL['collection']) \
            .filterDate(lookback_start, end) \
            .filterBounds(region) \
            .select(GEEDatasets.FIRE_THERMAL['bands'])
        
        # Land Surface Temperature
        collections['lst'] = ee.ImageCollection(GEEDatasets.LST['collection']) \
            .filterDate(lookback_start, end) \
            .filterBounds(region) \
            .select(GEEDatasets.LST['bands'])
        
        # Vegetation Index
        collections['ndvi'] = ee.ImageCollection(GEEDatasets.VEGETATION['collection']) \
            .filterDate(lookback_start, end) \
            .filterBounds(region) \
            .select(GEEDatasets.VEGETATION['bands'])
        
        # Weather
        collections['weather'] = ee.ImageCollection(GEEDatasets.WEATHER['collection']) \
            .filterDate(lookback_start, end) \
            .filterBounds(region) \
            .select(GEEDatasets.WEATHER['bands'])
        
        # Soil Moisture (Optional)
        if self.config.include_soil_moisture:
            collections['soil'] = ee.ImageCollection(GEEDatasets.SOIL_MOISTURE['collection']) \
                .filterDate(lookback_start, end) \
                .filterBounds(region) \
                .select(GEEDatasets.SOIL_MOISTURE['bands'])
        
        return collections
    
    def _safe_reduce(
        self,
        collection: ee.ImageCollection,
        reducer: str,
        band_names: list[str],
        default_value: float = 0
    ) -> ee.Image:
        """
        Safely reduce an image collection, handling empty collections.
        
        Args:
            collection: Image collection to reduce
            reducer: Reduction method ('mean', 'max', 'sum')
            band_names: Expected band names for the output
            default_value: Default value if collection is empty
            
        Returns:
            ee.Image: Reduced image or default image if collection is empty
        """
        # Create fallback image with default values
        fallback = ee.Image.constant([default_value] * len(band_names)) \
            .rename(band_names)
        
        # Check if collection is empty and apply appropriate reducer
        if reducer == 'mean':
            reduced = collection.mean()
        elif reducer == 'max':
            reduced = collection.max()
        elif reducer == 'sum':
            reduced = collection.sum()
        else:
            reduced = collection.mean()
        
        # Return reduced image if it has bands, otherwise return fallback
        return ee.Image(ee.Algorithms.If(
            collection.size().gt(0),
            reduced,
            fallback
        ))
    
    def _create_weekly_composite(
        self,
        week_start: ee.Date,
        collections: dict[str, ee.ImageCollection]
    ) -> ee.Image:
        """
        Create a feature composite for one week.
        
        Args:
            week_start: Start date of the week
            collections: Pre-filtered dynamic collections
            
        Returns:
            ee.Image: Multi-band composite with all features
        """
        week_end = week_start.advance(7, 'day')
        lookback_30 = week_start.advance(-30, 'day')
        
        static = self._load_static_datasets()
        threshold = self.config.fire_confidence_threshold
        
        # TARGET: Fire Occurrence 
        # NOTE: MOD14A1 is 8-day composite, so expand window by ±4 days to align
        # FireMask values: 7=low confidence, 8=nominal, 9=high confidence
        # Using gte(7) to include all fire classes (many Uttarakhand fires are class 7)
        fires_weekly = collections['fires'].filterDate(
            week_start.advance(-4, 'day'),
            week_end.advance(4, 'day')
        ).select('FireMask')
        fire_raw = self._safe_reduce(fires_weekly, 'max', ['FireMask'], 0)
        fire_occurred = fire_raw.unmask(0).gte(threshold).rename('fire_occurred')
        
        # VEGETATION FEATURES (with safe handling for empty collections)
        ndvi_weekly_coll = collections['ndvi'].filterDate(week_start, week_end)
        ndvi_raw = self._safe_reduce(ndvi_weekly_coll, 'mean', ['NDVI', 'EVI'], 0)
        ndvi_weekly = ndvi_raw.multiply(0.0001)
        
        ndvi_30day_coll = collections['ndvi'].filterDate(lookback_30, week_start)
        ndvi_30day_raw = self._safe_reduce(ndvi_30day_coll, 'mean', ['NDVI', 'EVI'], 0)
        ndvi_30day = ndvi_30day_raw.multiply(0.0001).select('NDVI').rename('ndvi_30day_mean')
        
        # TEMPERATURE FEATURES (with safe handling)
        lst_weekly_coll = collections['lst'].filterDate(week_start, week_end)
        lst_raw = self._safe_reduce(lst_weekly_coll, 'mean', ['LST_Day_1km', 'LST_Night_1km'], 0)
        lst_weekly = lst_raw.multiply(0.02)
        
        lst_day_max_coll = collections['lst'].filterDate(week_start, week_end).select('LST_Day_1km')
        lst_day_max_raw = self._safe_reduce(lst_day_max_coll, 'max', ['LST_Day_1km'], 0)
        lst_day_max = lst_day_max_raw.multiply(0.02).rename('lst_day_max')
        
        # WEATHER FEATURES (with safe handling)
        weather_weekly = collections['weather'].filterDate(week_start, week_end)
        
        temp_max_coll = weather_weekly.select('temperature_2m_max')
        temp_max = self._safe_reduce(temp_max_coll, 'max', ['temp_max'], 273.15)
        
        precip_7day_coll = weather_weekly.select('total_precipitation_sum')
        precip_7day_raw = self._safe_reduce(precip_7day_coll, 'sum', ['total_precipitation_sum'], 0)
        precip_7day = precip_7day_raw.multiply(1000).rename('precip_7day')
        
        precip_30day_coll = collections['weather'].filterDate(lookback_30, week_start).select('total_precipitation_sum')
        precip_30day_raw = self._safe_reduce(precip_30day_coll, 'sum', ['total_precipitation_sum'], 0)
        precip_30day = precip_30day_raw.multiply(1000).rename('precip_30day')
        
        # Wind speed from U and V components (with safe handling)
        wind_u_coll = weather_weekly.select('u_component_of_wind_10m')
        wind_v_coll = weather_weekly.select('v_component_of_wind_10m')
        wind_u = self._safe_reduce(wind_u_coll, 'mean', ['u_component_of_wind_10m'], 0)
        wind_v = self._safe_reduce(wind_v_coll, 'mean', ['v_component_of_wind_10m'], 0)
        wind_speed = wind_u.pow(2).add(wind_v.pow(2)).sqrt().rename('wind_speed')
        
        # === SOIL MOISTURE (Optional) ===
        if self.config.include_soil_moisture and 'soil' in collections:
            soil_moisture = collections['soil'] \
                .filterDate(week_start, week_end) \
                .mean() \
                .rename('soil_moisture')
        else:
            soil_moisture = ee.Image.constant(0).rename('soil_moisture')
        
        # === COMBINE ALL BANDS ===
        composite = fire_occurred \
            .addBands(ndvi_weekly) \
            .addBands(ndvi_30day) \
            .addBands(lst_weekly) \
            .addBands(lst_day_max) \
            .addBands(temp_max) \
            .addBands(precip_7day) \
            .addBands(precip_30day) \
            .addBands(wind_speed) \
            .addBands(soil_moisture) \
            .addBands(static['slope']) \
            .addBands(static['aspect']) \
            .addBands(static['elevation']) \
            .addBands(static['population']) \
            .addBands(static['land_cover']) \
            .addBands(static['burned_last_5yr']) \
            .addBands(static['burn_count_5yr']) \
            .addBands(static['years_since_last_burn']) \
            .set({
                'week_start': week_start.format('YYYY-MM-dd'),
                'week_end': week_end.format('YYYY-MM-dd'),
                'system:time_start': week_start.millis()
            })
        
        return composite
    
    def _create_weekly_collection(
        self, 
        start_date: str, 
        end_date: str
    ) -> ee.ImageCollection:
        """
        Create weekly composites for a date range.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            ee.ImageCollection: Collection of weekly composite images
        """
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        
        # Pre-load collections once for the entire period
        collections = self._load_dynamic_collections(start, end)
        
        # Calculate number of weeks
        num_weeks = end.difference(start, 'week').round()
        weeks = ee.List.sequence(0, num_weeks.subtract(1))
        
        def process_week(week_num: ee.Number) -> ee.Image:
            week_num = ee.Number(week_num)
            week_start = start.advance(week_num, 'week')
            return self._create_weekly_composite(week_start, collections)
        
        return ee.ImageCollection(weeks.map(process_week))
    
    def _sample_image(self, image: ee.Image) -> ee.FeatureCollection:
        """
        Sample pixels from a weekly composite with stratification.
        
        Stratified sampling:
        - More samples from fire pixels (rare class)
        - Fewer samples from non-fire pixels (common class)
        - Forest areas only
        
        Args:
            image: Weekly composite image
            
        Returns:
            ee.FeatureCollection: Sampled points with features
        """
        region = self._load_region()
        
        # Apply forest mask
        masked_image = image.updateMask(self._forest_mask)
        
        # Separate fire and non-fire pixels
        fire_pixels = masked_image.updateMask(
            image.select('fire_occurred').eq(1)
        )
        
        non_fire_pixels = masked_image.updateMask(
            image.select('fire_occurred').eq(0)
        )
        
        # Sample fire pixels (higher density - fires are rare)
        fire_count = fire_pixels.reduceRegion(
            reducer= ee.Reducer.count(),
            geometry= region,
            scale= self.config.sample_scale,
            maxPixels= 1e9
        ).get('fire_occurred')

        fire_samples = ee.FeatureCollection(
            ee.Algorithms.If(
                ee.Number(fire_count).gt(0),
                fire_pixels.sample(
                    region=region,
                    scale=self.config.sample_scale,
                    numPixels=self.config.samples_per_week_fire,
                    seed=42,
                    geometries=True,
                    dropNulls=True
                ),
                ee.FeatureCollection([])
            )
        )
        
        # Sample non-fire pixels (coarser to save EECU)
        non_fire_samples = non_fire_pixels.sample(
            region=region,
            scale=self.config.sample_scale * 2,
            numPixels=self.config.samples_per_week_nofire,
            seed=42,
            geometries=True
        )
        
        return fire_samples.merge(non_fire_samples)
    
    def _process_date_range(
        self, 
        start_date: str, 
        end_date: str
    ) -> ee.FeatureCollection:
        """
        Process a single date range: create composites and sample.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            ee.FeatureCollection: All samples from this period
        """
        logger.info(f"Processing: {start_date} to {end_date}")
        
        weekly_images = self._create_weekly_collection(start_date, end_date)
        samples = weekly_images.map(self._sample_image).flatten()
        
        return samples
    
    def export_to_drive(
        self, 
        samples: ee.FeatureCollection,
        description: Optional[str] = None
    ) -> ee.batch.Task:
        """
        Export samples to Google Drive as CSV.
        
        Args:
            samples: Feature collection to export
            description: Export task description (optional)
            
        Returns:
            ee.batch.Task: Export task (not started)
        """
        task = ee.batch.Export.table.toDrive(
            collection=samples,
            description=description or self.config.export_description,
            folder=self.config.export_folder,
            fileFormat='CSV',
            selectors=GEEDatasets.EXPORT_PROPERTIES
        )
        
        return task
    
    def export_to_local(
        self,
        samples: ee.FeatureCollection,
        output_path: str,
        batch_size: int = 5000
    ) -> str:
        """
        Export samples to local CSV file.
        
        Uses server-side batching to handle large datasets within GEE limits.
        
        Args:
            samples: Feature collection to export
            output_path: Local file path for CSV output
            batch_size: Number of features per batch (default 5000)
            
        Returns:
            str: Path to saved CSV file
        """
        logger.info(f"Exporting to local file: {output_path}")
        
        # Get total count
        total_count = samples.size().getInfo()
        logger.info(f"Total samples to export: {total_count}")
        
        if total_count == 0:
            logger.warning("No samples to export!")
            return output_path
        
        all_data = []
        
        # Convert to list and process in batches
        sample_list = samples.toList(total_count)
        
        num_batches = (total_count + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_count)
            
            logger.info(f"Fetching batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")
            
            # Get batch as FeatureCollection
            batch = ee.FeatureCollection(sample_list.slice(start_idx, end_idx))
            
            # Get feature info
            batch_info = batch.getInfo()
            
            for feature in batch_info['features']:
                props = feature['properties']
                # Add coordinates from geometry if available
                if feature.get('geometry') and feature['geometry'].get('coordinates'):
                    coords = feature['geometry']['coordinates']
                    props['longitude'] = coords[0]
                    props['latitude'] = coords[1]
                all_data.append(props)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
        return output_path
    
    def run_local(self, output_path: str) -> str:
        """
        Run the extraction pipeline and save locally.
        
        Args:
            output_path: Local path to save CSV file
            
        Returns:
            str: Path to saved file
        """
        logger.info(f"Starting {self.config.region_name} fire risk data extraction (local)")
        
        # Load prerequisites
        self._load_region()
        self._load_static_datasets()
        
        # Process all date ranges
        all_samples = []
        for idx, (start_date, end_date) in enumerate(self.config.date_ranges, 1):
            logger.info(f"[{idx}/{len(self.config.date_ranges)}] {start_date} to {end_date}")
            samples = self._process_date_range(start_date, end_date)
            all_samples.append(samples)
        
        # Merge all samples
        logger.info("Merging all samples...")
        final_samples = all_samples[0]
        for samples in all_samples[1:]:
            final_samples = final_samples.merge(samples)
        
        # Export locally
        return self.export_to_local(final_samples, output_path)
    
    def run(self, start_task: bool = True) -> ee.batch.Task:
        """
        Run the full extraction pipeline.
        
        Steps:
        1. Load region and static datasets
        2. Process each date range
        3. Merge all samples
        4. Export to Google Drive
        
        Args:
            start_task: Whether to start the export task automatically
            
        Returns:
            ee.batch.Task: Export task
        """
        logger.info(f"Starting {self.config.region_name} fire risk data extraction")
        
        # Load prerequisites
        self._load_region()
        self._load_static_datasets()
        
        # Process all date ranges
        all_samples = []
        for idx, (start_date, end_date) in enumerate(self.config.date_ranges, 1):
            logger.info(f"[{idx}/{len(self.config.date_ranges)}] {start_date} to {end_date}")
            samples = self._process_date_range(start_date, end_date)
            all_samples.append(samples)
        
        # Merge all samples
        logger.info("Merging all samples...")
        final_samples = all_samples[0]
        for samples in all_samples[1:]:
            final_samples = final_samples.merge(samples)
        
        # Create export task
        logger.info("Preparing export...")
        task = self.export_to_drive(final_samples)
        
        if start_task:
            task.start()
            logger.info(f"Export started: {self.config.export_description}")
            logger.info(f"Destination: Google Drive/{self.config.export_folder}/")
            logger.info(f"Task ID: {task.id}")
            logger.debug(f"Monitor at: https://code.earthengine.google.com/tasks")
        
        return task
    
    def monitor_task(self, task: ee.batch.Task, check_interval: int = 30) -> dict:
        """
        Monitor an export task until completion.
        
        Args:
            task: The export task to monitor
            check_interval: Seconds between status checks
            
        Returns:
            dict: Final task status
        """
        logger.info("Monitoring export progress...")
        
        last_state = None
        start_time = time.time()
        
        while task.active():
            status = task.status()
            state = status['state']
            
            if state != last_state:
                elapsed = time.time() - start_time
                logger.info(f"[{elapsed/60:.1f} min] Status: {state}")
                last_state = state
            
            time.sleep(check_interval)
        
        final_status = task.status()
        elapsed_total = time.time() - start_time
        
        logger.info(f"Export finished - State: {final_status['state']}, Time: {elapsed_total/60:.1f} min")
        
        if final_status['state'] == 'COMPLETED':
            logger.info(f"Output: Google Drive/{self.config.export_folder}/{self.config.export_description}.csv")
        else:
            logger.error("Export failed or cancelled")
            if 'error_message' in final_status:
                logger.error(f"Error: {final_status['error_message']}")
        
        return final_status



def process_new_data(
    start_date: str, 
    end_date: str, 
    output_name: str,
    config: Optional[FireDataConfig] = None
) -> ee.batch.Task:
    """
    Process new satellite data when it becomes available.
    
    Convenience function for processing additional time periods
    without re-running the full pipeline.
    
    Usage:
        task = process_new_data('2025-03-01', '2025-06-30', 'fire_data_2025')
        task.start()
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_name: Export file name
        config: Optional custom configuration
        
    Returns:
        ee.batch.Task: Export task (call .start() to begin)
    """
    cfg = config or FireDataConfig()
    cfg.date_ranges = [(start_date, end_date)]
    cfg.export_description = output_name
    
    extractor = FireDataExtractor(cfg)
    samples = extractor._process_date_range(start_date, end_date)
    task = extractor.export_to_drive(samples, output_name)
    
    logger.info(f"Task created: {output_name}")
    logger.info("Call task.start() to begin export")
    
    return task



if __name__ == '__main__':
    # Create configuration (modify as needed)
    config = FireDataConfig(
        region_name='Uttarakhand',
        date_ranges=[
            ('2023-03-01', '2023-06-30'),
            ('2024-03-01', '2024-06-30')
        ],
        samples_per_week_fire=8000,
        samples_per_week_nofire=4000,
        include_population=True,
        include_soil_moisture=False,  # Disabled to save EECU
        export_folder='HackEarth_Fire',
        export_description='uttarakhand_fire_data'
    )
    
    # Run extraction and save locally
    extractor = FireDataExtractor(config)
    output_path = Path(__file__).parent.parent / 'dataset' / 'fire_data' / 'uttarakhand_fire_data.csv'
    extractor.run_local(str(output_path))
    
    # Optional: To export to Google Drive instead, use:
    # task = extractor.run(start_task=True)
    # extractor.monitor_task(task)