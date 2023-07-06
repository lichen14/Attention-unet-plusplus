%preprocess_CHAOS����ɸѡԤ����.m;
Path = 'C:\Users\lichen\Desktop\CHAOS\dataset-dsb\test\mask\';                   % �������ݴ�ŵ��ļ���·��
File = dir(fullfile(Path,'*.png'));  % ��ʾ�ļ��������з��Ϻ�׺��Ϊ.txt�ļ���������Ϣ
FileNames = {File.name}';            % ��ȡ���Ϻ�׺��Ϊ.txt�������ļ����ļ�����ת��Ϊn��1��
Length_Names = size(FileNames,1);    % ��ȡ����ȡ�����ļ��ĸ���
figure(1)
for k = 1 : Length_Names
    % ����·�����ļ����õ��������ļ�·��
    K_Trace = strcat(Path, FileNames(k));
    % ��ȡ���ݣ���Ϊ������.txt��ʽ���ݣ�����ֱ����load()����)

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

% title('ȥ��������ע��ת','FontSize',12);