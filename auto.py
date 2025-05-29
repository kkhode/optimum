import pandas as pd
import argparse, sys, io, time, contextlib
import subprocess, os

from optimum.commands.export.onnx import parse_args_onnx

xlsxFileName = 'C:\\Users\\Local_Admin\\Downloads\\2025-02-10 - Model List v1.xlsx'
sheetName = 'PyTorchOnnxOpAnalysis'
currVenvPython = 'C:\\Users\\Local_Admin\\Downloads\\Kshitij\\Repos\\optimum-kkhode\\venv310\\Scripts\\python.exe'
modelStorePrefix = 'C:\\Users\\Local_Admin\\Downloads\\Kshitij\\Models\\PyTorchOnnxDynamo\\'
modeInfoXlsx = pd.read_excel(xlsxFileName, sheetName)
repoPathsCI = modeInfoXlsx.columns.get_loc('Model')
repoPaths = modeInfoXlsx.values[1:, repoPathsCI]

for rp in repoPaths:
    rpOnnxLoc = modelStorePrefix + rp.replace('/', '-')
    cmd = '%s %s %s %s %s %s %s %s' % (currVenvPython, '-m optimum.exporters.onnx.__main__ -m', rp, rpOnnxLoc, "--use-dynamo", "--debug-reports", "--opset", "23")
    if not os.path.exists(rpOnnxLoc): os.makedirs(rpOnnxLoc)
    stdoutFile = open(rpOnnxLoc + '\\stdout.log', 'w')
    stderrFile = open(rpOnnxLoc + '\\stderr.log', 'w')
    print('%s: Running %s' % (time.ctime(), cmd))
    # sys.argv = [__file__, "-m", rp, rpOnnxLoc]
    # exporterMain.main()
    op = subprocess.run(cmd, stdout=stdoutFile, stderr=stderrFile)
    print('%s: Finished running. RC: %s. stdout stored at: %s. stderr stored at: %s' % (time.ctime(), op.returncode, stdoutFile.name, stderrFile.name))
