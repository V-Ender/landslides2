import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.geometry as shg
import numpy as np
import rasterio
from rasterstats import zonal_stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def create_grid(base_shape, cell_size=150, out_path=''):
    x_min, y_min, x_max, y_max = base_shape.total_bounds

    grid_cells = []
    for x0 in np.arange(x_min, x_max + cell_size, cell_size):
        for y0 in np.arange(y_min, y_max + cell_size, cell_size):
            x1 = x0 - cell_size
            y1 = y0 - cell_size
            grid_cells.append(shg.box(x0, y0, x1, y1))

    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=base_shape.crs)
    grid_clip = gpd.clip(grid, base_shape, True)
    grid_clip.to_file(out_path + 'grid_clip.shp')

    return grid, grid_clip


def get_landslides(grid, landslides, cover_pct=0.2):
    targets = []
    for cell in grid['geometry']:
        flag = False
        for ls in landslides['geometry']:
            intersection = cell.intersection(ls)
            if intersection.area >= cell.area * cover_pct:
                targets.append(1)
                flag = True
                break
        if not flag:
            targets.append(0)

    grid['y'] = targets

    return grid


def aspect_mean(x):
    n = len(x)
    sin_mean = np.divide(np.sum(np.sin(np.radians(x))), n)
    cos_mean = np.divide(np.sum(np.cos(np.radians(x))), n)
    mean = np.arctan2(sin_mean, cos_mean)

    return np.degrees(mean)


def get_zonal_stats(grid, out_path, dem_path):
    slope_path = out_path + 'slope.tif'
    aspect_path = out_path + 'aspect.tif'
    TRI_path = out_path + 'TRI.tif'

    slopes = []
    aspects = []
    TRI = []
    elevations = []
    for i in range(len(grid)):
        cell = grid['geometry'].iloc[i]
        slopes.append(zonal_stats(cell, slope_path, stats=['mean'])[0]['mean'])
        aspects.append(zonal_stats(cell, aspect_path, stats=[], add_stats={'aspect_mean': aspect_mean})[0]['aspect_mean'])
        TRI.append(zonal_stats(cell, TRI_path, stats=['mean'])[0]['mean'])
        elevations.append(zonal_stats(cell, dem_path, stats=['mean'])[0]['mean'])


    grid['slope'] = slopes
    grid['aspect'] = aspects
    grid['TRI'] = TRI
    grid['elevation'] = elevations
    grid = grid.dropna()

    return grid


def load_dem(dem_path, out_path):
    dem = gdal.Open(dem_path)
    slope = gdal.DEMProcessing(out_path + 'slope.tif', dem, 'slope')
    aspect = gdal.DEMProcessing(out_path + 'aspect.tif', dem, 'aspect')
    TRI = gdal.DEMProcessing(out_path + 'TRI.tif', dem, 'TRI')


def transform_aspects(data):
    data['n'] = [1 if data['aspect'].iloc[i] < 22.5 or data['aspect'].iloc[i] >= 337.5 else 0 for i in range(data.shape[0])]
    data['ne'] = [1 if 22.5 <= data['aspect'].iloc[i] < 67.5 else 0 for i in range(data.shape[0])]
    data['e'] = [1 if 67.5 <= data['aspect'].iloc[i] < 112.5 else 0 for i in range(data.shape[0])]
    data['se'] = [1 if 112.5 <= data['aspect'].iloc[i] < 157.5 else 0 for i in range(data.shape[0])]
    data['s'] = [1 if 157.5 <= data['aspect'].iloc[i] < 202.5 else 0 for i in range(data.shape[0])]
    data['sw'] = [1 if 205.5 <= data['aspect'].iloc[i] < 247.5 else 0 for i in range(data.shape[0])]
    data['w'] = [1 if 247.5 <= data['aspect'].iloc[i] < 292.5 else 0 for i in range(data.shape[0])]
    data['nw'] = [1 if 292.5 <= data['aspect'].iloc[i] < 337.5 else 0 for i in range(data.shape[0])]

    return data.drop(columns=['aspect'])


def logistic_regression(data_landslides, data_nonlandslides, num_samples=100):
    C_arr = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
    best_model = LogisticRegression()
    best_score = 0.
    num_points = len(data_landslides)

    for _ in range(num_samples):
        nl_sampled = data_nonlandslides.sample(n=num_points)

        data = pd.concat([data_landslides, nl_sampled])
        data = data.reset_index()
        data = data.iloc[np.random.permutation(len(data))]
        y = data['y']

        data = data.drop(columns=['FID', 'y', 'geometry', 'index'])

        for C in C_arr:
            model = LogisticRegression(random_state=1, penalty='l2', C=C, solver='lbfgs', max_iter=1000)
            score = np.mean(cross_val_score(model, data, y, cv=5, scoring='roc_auc'))
            if score > best_score:
                print('Best ROC_AUC:', score, C)
                best_model = model
                best_score = score
                best_model.fit(data, y)
                print('Accuracy:', best_model.score(data, y))

    return best_model


def create_map(grid, model, out_path):
    data = grid.drop(columns=['y', 'geometry', 'FID'])
    predictions = model.predict(data)
    grid['pred'] = predictions
    grid = grid[['geometry', 'pred']]
    landslides_grid = grid[grid['pred'] == 1]
    nonlandslides_grid = grid[grid['pred'] == 0]

    landslides_grid.to_file(out_path + 'landslides_grid.shp')
    nonlandslides_grid.to_file(out_path + 'nonlandslides_grid.shp')
    show_results(out_path)


def show_results(out_path):
    nonlandslides_grid = gpd.read_file(out_path + 'nonlandslides_grid.shp')
    landslides_grid = gpd.read_file(out_path + 'landslides_grid.shp')

    map = pd.concat([nonlandslides_grid, landslides_grid], ignore_index=True)
    map.plot(column='pred')
    plt.show()


def process(dem_path, shape_path, landslides_path, cover_pct=0.2, num_samples=100, out_path=''):
    load_dem(dem_path, out_path)
    print('Dem loaded, rasters created')

    base_shape = gpd.read_file(shape_path)
    create_grid(base_shape, cell_size=150, out_path=out_path)
    print('Grid created')

    grid = gpd.read_file(out_path + 'grid_clip.shp')
    grid = get_zonal_stats(grid, out_path, dem_path)
    print('Raster values extracted to grid')
    
    landslides = gpd.read_file(landslides_path)
    grid = get_landslides(grid, landslides, cover_pct=cover_pct)
    grid = transform_aspects(grid)
    grid_landslides = grid[grid['y'] == 1]
    grid_nonlandslides = grid[grid['y'] == 0]

    print('Training...')
    model = logistic_regression(grid_landslides, grid_nonlandslides, num_samples)

    create_map(grid, model, out_path)
    print('Finished')

