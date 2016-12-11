% location of color images within data folder
image_folder = 'val_256'; % change this parameter to the relevant folder
file_path = strcat('data/', image_folder, '/');

files = dir(strcat(file_path, '*.jpg'));

% location of folder generated bw images
bw_file_path = strcat('data/', image_folder, '_bw/');
mkdir(bw_file_path);

for i = 1 : length(files)
    filename = strcat(file_path, files(i).name);
    I = imread(filename);
    % check if original image is rgb image
    if size(I,3) == 3
        bw = rgb2gray(I);
        imwrite(bw, strcat(bw_file_path, files(i).name));
    end
end
