import os
import shutil
import sys
import subprocess
import json
from PIL import Image, ImageChops

def chdir(dst):
    oldDir = os.getcwd()
    os.chdir(dst)
    return oldDir

def run_command(cmd):
    print(' '.join(cmd))
    ret = subprocess.run(cmd, check=True)

def run():
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    msbuild = R'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'

    # ----------------------------------------------------------------
    # Unit tests

    sln = os.path.abspath(os.path.join(scriptDir, R'..\tests\optixu_unit_tests.sln'))
    config = 'Release'
    exe = os.path.abspath(os.path.join(scriptDir, 'x64', config, 'optixu_tests.exe'))

    # Clean
    cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
    cmd += [sln]
    run_command(cmd)

    # Build
    cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
    cmd += [sln]
    run_command(cmd)

    print('Run unit tests')
    cmd = [exe]
    run_command(cmd)

    # END: Unit tests
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    # Sample image tests

    sln = os.path.abspath(os.path.join(scriptDir, R'..\samples\optixu_samples.sln'))
    refImgDir = os.path.abspath(os.path.join(scriptDir, R'ref_images'))

    with open(os.path.join(scriptDir, R'tests.json')) as f:
        tests = json.load(f)

    configs = ['Debug', 'Release']

    # Build
    for config in configs:
        # Clean
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
        cmd += [sln]
        run_command(cmd)

        # Build
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
        cmd += [sln]
        run_command(cmd)

    # Run tests
    results = {}
    for config in configs:
        resultsPerConfig = {}
        outDir = os.path.abspath(os.path.join(scriptDir, R'..\samples\x64', config))
        for test in tests:
            testName = test['sample']
            testDir = test['sample']
            exeName = test['sample'] + '.exe'

            print('Run ' + testName + ':')

            oldDir = chdir(os.path.join(scriptDir, R'..\samples', testDir))
            exe = os.path.join(outDir, exeName)
            cmd = [exe]
            if 'options' in test:
                cmd.append(test['options'])
            run_command(cmd)

            # RGBAでdiffをとると差が無いことになってしまう。
            testImgPath = test['image']
            refImgPath = os.path.join(refImgDir, testDir, 'reference.png')

            if os.path.exists(testImgPath) and os.path.exists(refImgPath):
                img = Image.open(testImgPath).convert('RGB')
                refImg = Image.open(refImgPath).convert('RGB')
                diffImg = ImageChops.difference(img, refImg)
                diffBBox = diffImg.getbbox()
                if diffBBox is None:
                    numDiffPixels = 0
                else:
                    numDiffPixels = sum(x != (0, 0, 0) for x in diffImg.crop(diffBBox).getdata())
            else:
                numDiffPixels = -1

            resultsPerConfig[testName] = {
                "success": numDiffPixels == 0,
                "numDiffPixels": numDiffPixels
            }

            for output in test['outputs']:
                if not os.path.exists(output):
                    continue
                if config == 'Release' and output == testImgPath and numDiffPixels != 0:
                    shutil.move(testImgPath, os.path.join(refImgDir, testDir))
                else:
                    os.remove(output)

            chdir(oldDir)
        
        results[config] = resultsPerConfig
    
    # Show results
    for config in configs:
        print('Test Results for ' + config + ':')
        resultsPerConfig = results[config]
        numSuccesses = 0
        for test, result in resultsPerConfig.items():
            print(test, result)
            if result['success']:
                numSuccesses += 1
        print('Successes: {}/{}, All Success: {}'.format(
            numSuccesses, len(resultsPerConfig), numSuccesses == len(resultsPerConfig)))
        print()

    # END: Sample image tests
    # ----------------------------------------------------------------

    return 0

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(e)
        