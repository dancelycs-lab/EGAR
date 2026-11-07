#!/usr/bin/env python3
"""Install the pinned public NMPC/iLQR sources and native libraries locally."""
import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
PEND_COMMIT='1e8544e67f3da72a51ee52fbbb278e131ce2a43f'
ACADOS_COMMIT='43354449192aa51557abda37d1bc17d907077354'

def run(args,cwd=None,env=None):
    print('+',shlex.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,env=env,check=True)

def checkout(url,path,commit,sparse=None):
    if not path.exists():
        run(['git','clone','--filter=blob:none','--no-checkout',url,path])
        if sparse:run(['git','sparse-checkout','set',*sparse],cwd=path)
        run(['git','checkout',commit],cwd=path)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    if current!=commit:raise RuntimeError(f'{path}: expected {commit}, found {current}. Use a separate installation directory.')

def build_ilqr(pend):
    from setuptools import Extension
    from Cython.Build import cythonize
    import numpy as np
    cpp=pend/'src/cpp';obj=cpp/'controllers/ilqr/obj';obj.mkdir(parents=True,exist_ok=True)
    prefix=Path(os.environ.get('CONDA_PREFIX',sys.prefix))
    eigen=next((p for p in [prefix/'include/eigen3',Path('/usr/include/eigen3'),Path('/usr/local/include/eigen3')] if (p/'Eigen').exists()),None)
    if eigen is None:raise RuntimeError('Eigen headers missing. Install libeigen3-dev or conda-forge eigen.')
    cxx=shlex.split(os.environ.get('CXX','g++'));ar=os.environ.get('AR','ar')
    sources=['model/src/dp_plant.cpp','simulator/src/simulator.cpp','controllers/ilqr/src/ilqr.cpp','controllers/ilqr/src/ilqr_mpc.cpp']
    objects=[]
    for source in sources:
        target=obj/(Path(source).stem+'.o')
        run([*cxx,'-std=c++17','-O3','-fPIC','-I'+str(eigen),'-I'+str(prefix/'include'),'-c',cpp/source,'-o',target])
        objects.append(target)
    library=obj/'libilqr.a'
    library.unlink(missing_ok=True)
    run([ar,'rcs',library,*objects])
    # Build only the iLQR wrapper, not unrelated grid controllers.
    # Its Cython directives compile the two controller sources; the archive
    # supplies the plant/simulator objects used by those sources.
    py=cpp/'python';old=Path.cwd();os.chdir(py)
    try:
        ext=Extension('cppilqr',['cppilqr.pyx'],libraries=['ilqr','yaml-cpp'],
            library_dirs=[str(obj),str(prefix/'lib')],
            runtime_library_dirs=[str(prefix/'lib')],
            include_dirs=[str(obj),str(eigen),str(prefix/'include'),np.get_include()],language='c++')
        from setuptools import Distribution
        dist=Distribution({'name':'cppilqr','ext_modules':cythonize([ext],compiler_directives={'language_level':3})})
        cmd=dist.get_command_obj('build_ext');cmd.inplace=True;cmd.ensure_finalized();cmd.run()
    finally:os.chdir(old)

def build_acados(path,jobs):
    run(['git','submodule','update','--init','--recursive'],cwd=path)
    run(['cmake','-S',path,'-B',path/'build','-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_INSTALL_PREFIX='+str(path),'-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
         '-DCMAKE_C_FLAGS=-D_POSIX_C_SOURCE=200809L','-DBLASFEO_TARGET=X64_AUTOMATIC',
         '-DHPIPM_TARGET=X64_AUTOMATIC'])
    run(['cmake','--build',path/'build','--parallel',str(jobs)])
    run(['cmake','--install',path/'build'])
    renderer=path/'bin/t_renderer';renderer.parent.mkdir(exist_ok=True)
    if not renderer.exists():
        url='https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux'
        with urllib.request.urlopen(url) as response,renderer.open('wb') as stream:shutil.copyfileobj(response,stream)
        renderer.chmod(0o755)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--only',choices=['all','ilqr','acados'],default='all')
    a=ap.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','AMD64']:
        ap.error('This native build recipe targets Linux x86_64.')
    external=ROOT/'external';external.mkdir(exist_ok=True)
    pend=external/'double_pendulum'
    checkout('https://github.com/dfki-ric-underactuated-lab/double_pendulum.git',pend,PEND_COMMIT,['src/python','src/cpp'])
    if a.only in ['all','ilqr']:build_ilqr(pend)
    if a.only in ['all','acados']:
        acados=external/'acados';checkout('https://github.com/acados/acados.git',acados,ACADOS_COMMIT)
        build_acados(acados,a.jobs)
    print('Native baseline installation complete. Run: python reproduce.py all')

if __name__=='__main__':main()
