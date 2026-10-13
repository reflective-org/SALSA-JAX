"""Plot hourly native-bin distributions and time-size heatmaps without smoothing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(values):
    return np.where(values > 0, values, np.nan)


def _save(fig, output, stem, *, close=True):
    for suffix in ('png', 'svg'):
        fig.canvas.draw()
        fig.savefig(output/f'{stem}.{suffix}', dpi=190, bbox_inches='tight')
    if close:
        plt.close(fig)


def plot(run_directory, output):
    """Create plots, hourly numerical table and source-hashed plotting metadata."""
    run_directory, output = Path(run_directory), Path(output)
    source_names = ('jax.npz', 'fortran.npz', 'initial.npz', 'bins.csv', 'report.json')
    jax = dict(np.load(run_directory/'jax.npz'))
    native = dict(np.load(run_directory/'fortran.npz'))
    initial = dict(np.load(run_directory/'initial.npz'))
    report = json.loads((run_directory/'report.json').read_text())
    bins = np.genfromtxt(run_directory/'bins.csv', delimiter=',', names=True)
    time_s = np.asarray(jax['time_s'])
    if not np.array_equal(time_s, native['time_s']):
        raise ValueError('JAX and native saved times differ')
    if time_s.ndim != 1 or not len(time_s) or time_s[0] <= 0 or np.any(np.diff(time_s) <= 0):
        raise ValueError('Saved times must strictly increase from a positive first sample')
    fields = bins.dtype.names
    if fields != ('bin', 'lower_diameter_m', 'midpoint_diameter_m', 'upper_diameter_m'):
        raise ValueError('Unexpected bin-edge CSV schema')
    lower = bins['lower_diameter_m'][:10]*1e9
    upper = bins['upper_diameter_m'][:10]*1e9
    midpoint = bins['midpoint_diameter_m'][:10]*1e9
    if not np.allclose(lower[1:], upper[:-1], rtol=1e-12, atol=0):
        raise ValueError('Soluble nominal bin edges must be contiguous')
    edges = np.r_[lower, upper[-1]]
    widths = np.log10(upper/lower)
    time = np.r_[0.,time_s]/3600.
    numbers = {}
    for name,data in (('JAX',jax),('Fortran',native)):
        if data['number'].shape != (len(time_s),1,17) or initial['number'].shape != (1,17):
            raise ValueError('Expected one model cell and seventeen native bins')
        number = np.concatenate((initial['number'],data['number'][:,0]))/1e6
        if not np.isfinite(number).all() or np.any(number < 0):
            raise ValueError('Cannot plot negative or nonfinite particle number')
        if np.any(number[:,10:] != 0):
            raise ValueError('This soluble-only plot would omit populated insoluble bins')
        numbers[name] = number[:,:10]
    distributions = {name:number/widths for name,number in numbers.items()}
    duration = time[-1]
    hours = np.arange(int(np.floor(duration+1e-10))+1)
    indices = []
    for hour in hours:
        matches = np.flatnonzero(np.isclose(time,hour,rtol=0,atol=1e-9))
        if len(matches) != 1:
            raise ValueError(f'Exactly one saved state required at hour {hour}')
        indices.append(int(matches[0]))
    indices = np.array(indices)
    output.mkdir(parents=True,exist_ok=False)
    configuration = report.get('configuration',{})
    title = str(configuration.get('label',run_directory.name)).replace('_',' ')
    cmap = plt.get_cmap('viridis')
    hour_norm = Normalize(0,max(duration,1))
    all_positive = np.concatenate([values[values > 0] for values in distributions.values()])
    vmin = float(all_positive.min()) if len(all_positive) else 1.
    vmax = float(all_positive.max()) if len(all_positive) else 10.
    if vmin == vmax:
        vmin,vmax = vmin/2,vmax*2
    ylabel = r'$dN/d\log_{10}D_p$ (cm$^{-3}$)'
    fig, axes = plt.subplots(2,2,figsize=(13,8.6),sharex=True,layout='constrained')
    for axis,name in zip(axes[0],('Fortran','JAX')):
        for hour,index in zip(hours,indices):
            axis.stairs(_positive(distributions[name][index]),edges,color=cmap(hour_norm(hour)),linewidth=1.1)
        axis.set_title(f'{name}: every hour including initial state')
        axis.set_ylabel(ylabel)
        axis.set_yscale('log')
        axis.set_ylim(vmin*.75,vmax*1.4)
    overlay = axes[1,0]
    selected_positions = np.unique(np.rint(np.linspace(0,len(hours)-1,min(7,len(hours)))).astype(int))
    for position in selected_positions:
        hour,index = hours[position],indices[position]
        color = cmap(hour_norm(hour))
        overlay.stairs(_positive(distributions['JAX'][index]),edges,color=color,linewidth=1.7)
        overlay.stairs(_positive(distributions['Fortran'][index]),edges,color=color,linewidth=1.,linestyle='--')
    overlay.plot([],[],color='black',label='JAX',linewidth=1.7)
    overlay.plot([],[],color='black',label='Fortran',linewidth=1.,linestyle='--')
    overlay.legend(loc='best',fontsize=9)
    overlay.set_title('Overlay: '+', '.join(str(hours[p]) for p in selected_positions)+' h')
    overlay.set_ylabel(ylabel)
    overlay.set_yscale('log'); overlay.set_ylim(vmin*.75,vmax*1.4)
    delta = distributions['JAX']-distributions['Fortran']
    difference = axes[1,1]
    for hour,index in zip(hours,indices):
        difference.stairs(delta[index],edges,color=cmap(hour_norm(hour)),linewidth=1.1)
    difference.axhline(0,color='grey',linewidth=.7)
    difference.set_title('JAX − Fortran: every hour')
    difference.set_ylabel(r'Difference in $dN/d\log_{10}D_p$ (cm$^{-3}$)')
    difference.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    for axis in axes.ravel():
        axis.set_xscale('log');axis.set_xlim(edges[0],edges[-1])
        axis.grid(alpha=.2,which='major');axis.set_xlabel('Nominal dry particle diameter (nm)')
    fig.colorbar(plt.cm.ScalarMappable(norm=hour_norm,cmap=cmap),ax=axes.ravel().tolist(),label='Time (h)',shrink=.85)
    fig.suptitle(title+' — native-bin number distributions\n10 soluble bins; exact bin widths; zero counts are gaps on logarithmic axes',fontsize=13)
    _save(fig,output,'hourly-size-distributions-full-range',close=False)
    display_floor = 1e-3
    diameter_limits = (max(edges[0],3.),min(edges[-1],1000.))
    for axis in axes.ravel():
        axis.set_xlim(*diameter_limits)
    for axis in (*axes[0],overlay):
        axis.set_ylim(display_floor,vmax*1.4)
    fig.suptitle(title+' — native-bin number distributions\n3–1000 nm; floor: 10⁻³ cm⁻³; smaller values and zeros lie below this view',fontsize=13)
    _save(fig,output,'hourly-size-distributions')

    # Mesh cells are centered on saved times; boundary cells are clipped to the
    # observed interval. No temporal or size interpolation is applied.
    time_edges = np.r_[time[0],(time[:-1]+time[1:])/2,time[-1]]
    fig,axes = plt.subplots(1,3,figsize=(15,5.2),sharex=True,sharey=True,layout='constrained')
    norm = LogNorm(vmin=vmin,vmax=vmax)
    number_meshes = []
    number_cmap = plt.get_cmap('viridis').with_extremes(under='#d9d9d9',bad='white')
    for axis,name in zip(axes[:2],('Fortran','JAX')):
        mesh = axis.pcolormesh(time_edges,edges,np.ma.masked_less_equal(distributions[name].T,0),
                               norm=norm,cmap=number_cmap,shading='flat',rasterized=True)
        number_meshes.append(mesh)
        axis.set_title(name)
    fig.colorbar(mesh,ax=axes[:2].tolist(),label=ylabel,shrink=.85,extend='min')
    max_delta = float(np.max(np.abs(delta)))
    scale = max(max_delta,1e-20)
    difference_norm = SymLogNorm(linthresh=scale/100,vmin=-scale,vmax=scale,base=10)
    mesh = axes[2].pcolormesh(time_edges,edges,delta.T,norm=difference_norm,cmap='RdBu_r',
                            shading='flat',rasterized=True)
    axes[2].set_title('JAX − Fortran (absolute units)')
    fig.colorbar(mesh,ax=axes[2],label=r'Δ $dN/d\log_{10}D_p$ (cm$^{-3}$)',shrink=.85)
    for axis in axes:
        axis.set_yscale('log');axis.set_ylim(edges[0],edges[-1]);axis.set_xlim(0,duration)
        axis.set_xlabel('Time (h)')
    axes[0].set_ylabel('Nominal dry particle diameter (nm)')
    fig.suptitle(title+' — time–size distribution (“banana” plot)\nNative size bins and saved times; common logarithmic number scale; white means zero',fontsize=13)
    _save(fig,output,'banana-size-distributions-full-range',close=False)
    display_norm = LogNorm(vmin=display_floor,vmax=max(vmax,display_floor*10))
    for number_mesh in number_meshes:
        number_mesh.set_norm(display_norm)
    for axis in axes:
        axis.set_ylim(*diameter_limits)
    fig.suptitle(title+' — time–size distribution (“banana” plot)\n3–1000 nm; gray: 0 < dN/dlog₁₀D < 10⁻³ cm⁻³; white: zero; no smoothing or interpolation',fontsize=13)
    _save(fig,output,'banana-size-distributions')

    table=[]
    for hour,index in zip(hours,indices):
        for bin_index in range(10):
            table.append((hour,bin_index+1,lower[bin_index],midpoint[bin_index],upper[bin_index],
                          numbers['Fortran'][index,bin_index],numbers['JAX'][index,bin_index],
                          distributions['Fortran'][index,bin_index],distributions['JAX'][index,bin_index],delta[index,bin_index]))
    np.savetxt(output/'hourly-size-distributions.csv',np.asarray(table),delimiter=',',comments='',
               header='hour,bin,lower_diameter_nm,midpoint_diameter_nm,upper_diameter_nm,fortran_number_cm3,jax_number_cm3,fortran_dNdlog10D_cm3,jax_dNdlog10D_cm3,jax_minus_fortran_dNdlog10D_cm3')
    plotting_report = {'configuration':configuration,'source_sha256':{name:_sha(run_directory/name) for name in source_names},
                       'plotting_script_sha256':_sha(Path(__file__)),
                       'saved_times_including_initial':len(time),'hourly_times':hours.tolist(),
                       'diameter_coordinate':'nominal dry size-bin edges, not wet or per-bin mean particle diameter',
                       'number_normalization':'particles/cm3 divided by log10(upper diameter/lower diameter)',
                       'source_soluble_bins':list(range(1,11)),'omitted_insoluble_bins':list(range(11,18)),
                       'primary_diameter_limits_nm':list(diameter_limits),
                       'primary_display_floor_dNdlog10D_cm3':display_floor,
                       'primary_below_floor_display':'gray positive heatmap cells; below hourly y-axis limit',
                       'full_range_figures':'files with -full-range retain all positive number values and all soluble bin diameters',
                       'omitted_bins_verified_zero':True,'smoothing':False,'interpolation':False,
                       'heatmap_time_cells':'midpoint boundaries between saved times; endpoint cells clipped to simulation interval',
                       'zero_number_display':'masked white heatmap cells; gaps in logarithmic hourly curves',
                       'full_range_heatmap_number_limits_cm3':[vmin,vmax],
                       'primary_heatmap_number_limits_cm3':[display_floor,max(vmax,display_floor*10)],
                       'max_abs_number_difference_cm3':float(np.max(np.abs(numbers['JAX']-numbers['Fortran']))),
                       'max_abs_dNdlog10D_difference_cm3':max_delta,
                       'files_sha256':{path.name:_sha(path) for path in output.iterdir() if path.is_file()}}
    (output/'plot-report.json').write_text(json.dumps(plotting_report,indent=2)+'\n')
    return plotting_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh directory for figures and numerical table')
    args=parser.parse_args()
    report=plot(args.run_directory,args.output)
    print(json.dumps({key:report[key] for key in ('hourly_times','max_abs_number_difference_cm3','max_abs_dNdlog10D_difference_cm3')},indent=2))


if __name__ == '__main__':
    main()
