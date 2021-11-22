clear;

colors = {
    [1.0000, 0.0000, 0.0000]
    [0.7059, 0.5333, 0.8824]
    [0.8000, 0.8000, 0.1000]
    [0.9373, 0.6863, 0.1255]
    [0.0588, 0.6471, 0.6471]
    [0.0000, 1.0000, 1.0000]
    [0.0000, 0.0000, 1.0000]
    [0.7098, 0.2000, 0.3608]
    [0.7176, 0.5137, 0.4392]
    [0.7098, 0.2000, 0.3608]
    [0.4902, 0.0706, 0.6863]
    [0.0392, 0.4275, 0.2667]
    [0.4157, 0.5373, 0.0824]
    [0.5490, 0.5490, 0.4549]
};

lines = {'-','-','-','-','-','-','-','--','--','--','--','--','--','--'};

names = {
    'Ours'
    'DFF'
    'RCF'
    'CASENet'
    'BDCN'
    'OFNet'
    'DOOBNet'
    'DeepLabV3+'
    'HED'
    '*OFNet'
    '*DOOBNet'
    '*DeepLabV3+'
    'CED'
    'DexiNed'
};

years = {
    ' (2020)'
    ' (2019)'
    ' (2019)'
    ' (2018)'
    ' (2017)'
    ' (2017)'
    ' (2015)'
    ' (2015)'
};

edgesEvalPlot('eval_depth', names, colors, lines, years, true);
% saveas(gcf,'depth.png');
% close all
edgesEvalPlot('eval_normal', names, colors, lines, years, true);
% saveas(gcf,'normal.png');
% close all
edgesEvalPlot('eval_reflectance', names, colors, lines, years, true);
% saveas(gcf,'reflectance.png');
% close all
edgesEvalPlot('eval_illumination', names, colors, lines, years, true);
% saveas(gcf,'illumination.png');
% close all
