"""Deterministic Figure 1: complete DEM rectangle and a white ocean mask.

Display-only cartography. No model input, gauge coordinate or terrain-forcing
value is changed. Natural Earth 10m denotes 1:10 million, not 10-m resolution.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

CAPTION=('Figure 1. Sounding (red star) and rainfall-gauge (black circles) locations '
         'over Copernicus GLO-30 land surface elevation. The sea is white, delineated '
         'using the generalized Natural Earth 1:10 million land mask. Four DSM tiles '
         'provide continuous rectangular coverage; station coordinates are unchanged.')

TILES=['N07_00_E098_00','N07_00_E099_00','N08_00_E098_00','N08_00_E099_00']
POINTS={
    'Phuket Airport sounding':(98.3144,8.145),
    'Phuket':(98+24/60,7+53/60),
    'Takua Pa (Phang-nga)':(98+15/60,8+41/60),
    'Krabi':(98+58/60,8+6/60),
    'Nakhon Si Thammarat':(99+56/60,8+32/60),
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_rings(geometry):
    polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
    for polygon in polygons:
        for ring in polygon:yield np.asarray(ring)


def draw_station_map(base, output_directory, map_assets=None):
    base=Path(base);output_directory=Path(output_directory)
    assets=Path(map_assets) if map_assets else base/'data/external/figure1'
    output_directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for tile in TILES:
        name=f'Copernicus_DSM_COG_10_{tile}_DEM.tif'
        candidates=[assets/name,base/'data/external/dem'/name]
        path=next((p for p in candidates if p.exists()),None)
        if path is None:raise FileNotFoundError(f'Missing map tile: {name}')
        paths.append(path)
    # Right border stays within the true eastern tile edge, preserving every
    # station. The former extra 0.08-degree blank strip is not part of the frame.
    with rasterio.open(paths[-1]) as src:right=min(100.0,src.bounds.right)
    left,bottom,top=98.03,7.65,8.94
    width=1800;height=round(width*(top-bottom)/(right-left))
    transform=from_bounds(left,bottom,right,top,width,height)
    elevation=np.full((height,width),np.nan,dtype='float32')
    for path in paths:
        with rasterio.open(path) as src:
            piece=np.full_like(elevation,np.nan)
            reproject(source=rasterio.band(src,1),destination=piece,
                      src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
                      dst_transform=transform,dst_crs='EPSG:4326',dst_nodata=np.nan,
                      resampling=Resampling.average,num_threads=2)
            valid=np.isfinite(piece);elevation[valid]=piece[valid]
    uncovered=int((~np.isfinite(elevation)).sum())
    if uncovered:raise ValueError(f'Rectangle contains {uncovered} uncovered DEM display pixels')
    mask_path=assets/'ne_10m_land_map_region.geojson'
    if not mask_path.exists():mask_path=assets/'ne_10m_land.geojson'
    collection=json.loads(mask_path.read_text(encoding='utf-8'))
    geometries=[f['geometry'] for f in collection['features']]
    land=rasterize([(g,1) for g in geometries],out_shape=elevation.shape,
                   transform=transform,fill=0,dtype='uint8').astype(bool)
    if not land.any() or land.all():raise ValueError('Land/ocean mask is not valid for this coastal map')
    # Zero/negative land elevations are NOT automatically turned into sea.
    displayed=np.ma.array(elevation,mask=~land)
    cmap=plt.get_cmap('terrain').copy();cmap.set_bad('white')
    fig,ax=plt.subplots(figsize=(9,7),facecolor='white')
    fig.subplots_adjust(left=.09,right=.85,bottom=.13,top=.89)
    ax.set_facecolor('white')
    artist=ax.imshow(displayed,extent=[left,right,bottom,top],origin='upper',
                     vmin=0,vmax=1500,cmap=cmap,interpolation='nearest',zorder=1)
    for geometry in geometries:
        for ring in geometry_rings(geometry):
            ax.plot(ring[:,0],ring[:,1],color='#5e635d',lw=.45,zorder=2,clip_on=True)
    offsets={
        'Phuket Airport sounding':(-10,-20),
        'Phuket':(10,-13),
        'Takua Pa (Phang-nga)':(10,9),
        'Krabi':(10,10),
        'Nakhon Si Thammarat':(-10,13),
    }
    # Two-line sounding label stays within the map border.
    labels={'Phuket Airport sounding':'Phuket Airport\nsounding'}
    for name,(lon,lat) in POINTS.items():
        is_sounding='sounding' in name
        ax.scatter(lon,lat,s=135 if is_sounding else 52,
                   marker='*' if is_sounding else 'o',
                   c='#d62828' if is_sounding else 'black',
                   edgecolors='white',linewidths=1.1,zorder=5)
        align='right' if name=='Nakhon Si Thammarat' else 'left'
        ax.annotate(labels.get(name,name),(lon,lat),xytext=offsets[name],
                    textcoords='offset points',fontsize=9,ha=align,
                    bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=2),
                    zorder=6,annotation_clip=True)
    ax.set(xlim=(left,right),ylim=(bottom,top),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.set_aspect(1/math.cos(math.radians((bottom+top)/2)))
    ax.set_xticks(np.arange(98.25,100.001,.25));ax.set_yticks(np.arange(7.8,8.901,.2))
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(1.0);spine.set_color('black')
    ax.annotate('',xy=(.956,.965),xytext=(.956,.86),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',color='black',lw=1.6),zorder=7)
    ax.text(.956,.838,'N',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    scale_lat=7.72;scale_lon=98.11
    dx=50/(111.32*math.cos(math.radians(scale_lat)))
    ax.plot([scale_lon,scale_lon+dx],[scale_lat,scale_lat],color='black',lw=3,zorder=7)
    ax.text(scale_lon+dx/2,scale_lat+.021,'50 km',ha='center',fontsize=9,zorder=7)
    # Match the colour-bar height to the actual map frame.
    fig.canvas.draw();position=ax.get_position()
    colour_axes=fig.add_axes([position.x1+.025,position.y0,.024,position.height])
    colourbar=fig.colorbar(artist,cax=colour_axes,label='Surface elevation (m)')
    colourbar.ax.tick_params(labelsize=9)
    fig.suptitle('Station locations and surface elevation',fontsize=13,y=position.y1+.055)
    png=output_directory/'fig_station_map.png'
    fig.savefig(png,dpi=300,facecolor='white')
    fig.savefig(output_directory/'fig_station_map.pdf',facecolor='white')
    plt.close(fig)
    report={
        'scope':'Figure 1 display only; no model inputs or analysis results changed',
        'bounds':[left,bottom,right,top],'display_shape':[height,width],
        'dem_tile_count':len(paths),'uncovered_display_pixels':uncovered,
        'land_pixels':int(land.sum()),'sea_pixels':int((~land).sum()),
        'land_mask':'Natural Earth 1:10 million generalized land polygons; NOT 10-m resolution',
        'sea_colour':'#ffffff','land_elevation_range_m':[0,1500],
        'zero_elevation_does_not_define_sea':True,
        'points':{k:list(v) for k,v in POINTS.items()},'caption':CAPTION,
        'source_files':[{ 'path':str(p),'sha256':sha(p)} for p in paths+[mask_path]],
        'png':str(png),'png_sha256':sha(png),
    }
    (output_directory/'figure1_map_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_files','points']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-dir',type=Path,default=Path(os.environ.get('RAINFALL_PROJECT_DIR',Path(__file__).resolve().parent.parent)))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--map-assets',type=Path)
    args=parser.parse_args()
    draw_station_map(args.project_dir,args.output_dir or args.project_dir/'outputs/revision_20260907/figures',args.map_assets)
