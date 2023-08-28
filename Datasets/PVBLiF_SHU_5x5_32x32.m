clc;close all;clear;

dataset_path = 'xxx\SHU\'; % Set the dataset path here
savepath = 'xxx\PVBLiF_SHU_5x5_32x32\'; % Set the save path here

% for SHU dataset
load('SHU_all_info.mat');
load('SHU_all_mos.mat');
Distorted_sceneNum = 240; 

angRes = 5;             
patchsize = 32;  

for iScene = 1 : Distorted_sceneNum
    
    idx = 1;
    idx_s = 0;
    h5_savedir = [savepath, '\',SHU_all_info{1}{iScene}, '\',  SHU_all_info{2}{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, SHU_all_info{6}{iScene}];
    LF = load(dataPath).im2;
    
    LF = LF(:,:,2:434,2:624,:);
    hnumber = floor(433/patchsize);
    wnumber = floor(623/patchsize);
    hstart = floor((433-patchsize*hnumber)/2);
    wstart = floor((623-patchsize*wnumber)/2);
    LF = LF(:,:,hstart:hstart+hnumber*patchsize-1,wstart:wstart+wnumber*patchsize-1,:);
    total_patch_number = 247;

    [U, V, ~, ~, ~] = size(LF);
    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, :);
    [U, V, H, W, ~] = size(LF);
 
    dis_data_mirco = single(zeros(total_patch_number, U * patchsize, V * patchsize));

    dis_LF_VS = single(zeros(U, V, H, W));
    for u = 1 : U
        for v = 1 : V 
            dis_LF_VS(u,v,:,:) = VisualSaliency(squeeze(LF(u,v,:,:,:)));
        end
    end
    
    label = str2num(SHU_all_mos{iScene});
    all_VS_list = [];
    var_list = [];
    for h = 1 : patchsize : H
        for w = 1 : patchsize : W
            idx_s = idx_s + 1;            
            all_VS = [];
            PVBS_var = [];
            for u = 1 : U
                for v = 1 : V                        
                    temp_dis = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                    VS_Score = max(max(squeeze(dis_LF_VS(u, v, h : h+patchsize-1, w : w+patchsize-1))));
                    all_VS = [all_VS, VS_Score];
                    temp_dis = rgb2ycbcr(temp_dis);
                    temp_dis = squeeze(temp_dis(:,:,1));
                    dis_data_mirco(idx_s, u:angRes:U * patchsize, v:angRes:V * patchsize) = temp_dis;  
                    PVBS_var = [PVBS_var, var(double(temp_dis(:)))];
                end
            end  
            var_list(idx_s,1) = mean(PVBS_var);
            all_VS_list = [all_VS_list, mean(all_VS)];
        end
    end
    
    all_VS_list = all_VS_list';
    [var_list, index]  = sort(var_list);
    for i = 1:total_patch_number
        save_dis_data_mirco = squeeze(dis_data_mirco(index(i),:,:)); 
        save_dis_data_mirco_3D = single(zeros(patchsize, patchsize, angRes*angRes));
        for x = 1:angRes
            for y = 1:angRes
                save_dis_data_mirco_3D(:,:,(x-1)*angRes+y) = save_dis_data_mirco(x:angRes:angRes*patchsize,y:angRes:angRes*patchsize);
            end
        end
        SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
        h5create(SavePath_H5_name, '/dis_data', size(save_dis_data_mirco_3D), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/dis_data', single(save_dis_data_mirco_3D), [1,1,1], size(save_dis_data_mirco_3D));
        h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
        h5create(SavePath_H5_name, '/VS', size(all_VS_list(index(i))), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/VS', single(all_VS_list(index(i))), [1,1], size(all_VS_list(index(i))));
        idx = idx + 1;   
    end
end


