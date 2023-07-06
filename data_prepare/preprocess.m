%preprocess_CHAOS数据筛选预处理.m;
Path = 'C:\Users\lichen\Desktop\CHAOS\dataset-dsb\test\mask\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.png'));  % 显示文件夹下所有符合后缀名为.txt文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为.txt的所有文件的文件名，转换为n行1列
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数
figure(1)
for k = 1 : Length_Names
    % 连接路径和文件名得到完整的文件路径
    K_Trace = strcat(Path, FileNames(k));
    % 读取数据（因为这里是.txt格式数据，所以直接用load()函数)

    seg1=imread(K_Trace{1,1});
    imshow(seg1);
    
    [len,wid]=size(seg1);
    for i=1:len
        for j=1:wid
            if seg1(i,j)<70 && seg1(i,j)>55 
                seg1(i,j)=255;
            else
                seg1(i,j)=0;
            end
        end
    end
    for i=1:len
        for j=1:len
            if i>1 && i<len 
                if seg1(i-1,j)==0 && seg1(i,j)==255 && seg1(i+1,j)==0
                    seg1(i,j)=0;
                end
            end
        end
    end
    
        
    num= sum(sum(seg1>0));
    ratio = num/(len*wid);
    if ratio>0.009
        mask_Path=strrep(Path,'mask','mask-liver');
        old_imagefile_Path = strrep(Path,'mask','image');
        new_imagefile_Path = strrep(Path,'mask','image-liver');
        mask_Path=strcat(mask_Path,FileNames(k));
        file_name=strrep(FileNames(k),'mask','image');
        old_image_Path=strcat(old_imagefile_Path,file_name); 
        new_image_Path = strcat(new_imagefile_Path,file_name);
        image=imread(old_image_Path{1,1});
        new_path=mask_Path{1,1};
%         imshow(seg1);
        imwrite(seg1,new_path);
        imwrite(image,new_image_Path{1,1});
    end
end

% title('去除机器标注翻转','FontSize',12);