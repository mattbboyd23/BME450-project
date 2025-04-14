input_folder = '.';  % Directory with original images
output_folder = '.'; % Same folder, or change to save elsewhere

% Prompt for target dimensions
target_width = input('Enter target width in pixels: ');
target_height = input('Enter target height in pixels: ');

% Find all sample files like sampleX-#######.jpg
files = dir(fullfile(input_folder, 'sample*-???????.jpg'));
files = files(~contains({files.name}, '-resized'));

fprintf('Found %d sample images.\n', length(files));

for i = 1:length(files)
    fname = files(i).name;
    full_input_path = fullfile(input_folder, fname);

    % Read and resize
    img = imread(full_input_path);
    resized_img = imresize(img, [target_height, target_width]);

    % Create new filename: add "-resized" before .jpg
    [~, name, ~] = fileparts(fname);
    resized_name = strcat(name, '-resized.jpg');
    full_output_path = fullfile(output_folder, resized_name);

    % Save resized image
    imwrite(resized_img, full_output_path);

    fprintf('Saved: %s (%d × %d)\n', resized_name, target_width, target_height);
end

fprintf('All %d images resized and saved.',length(files));