% ---- annotate_samples.m ----

start_id = input('Enter starting sample ID: ');
end_id = input('Enter ending sample ID: ');

folder = '.';  % Folder with sample images
filename = 'outputs.xlsx';
T = readtable(filename);

labels = {'RH', 'RK', 'RA', 'LH', 'LK', 'LA'};
colors = repmat({'y'}, 1, 6);
regions = {
    [0, 0.5, 0, 0.5];     % RH
    [0, 0.5, 0.25, 0.75]; % RK
    [0, 0.5, 0.5, 1];     % RA
    [0.5, 1, 0, 0.5];     % LH
    [0.5, 1, 0.25, 0.75]; % LK
    [0.5, 1, 0.5, 1];     % LA
};

for sample_id = start_id:end_id
    files = dir(fullfile(folder, sprintf('sample%d-*.jpg', sample_id)));

    if isempty(files)
        warning('No file found for sample %d. Skipping...', sample_id);
        continue;
    end

    fname = files(1).name;

    % Extract patient ID
    tokens = regexp(fname, 'sample(\d+)-([\d]{7})\.jpg', 'tokens');
    if isempty(tokens)
        warning('Filename format invalid: %s. Skipping...', fname);
        continue;
    end
    tokens = tokens{1};
    patient_id = tokens{2};

    % Load image
    img = imread(fullfile(folder, fname));
    [h, w, ~] = size(img);
    x = zeros(6, 1);
    y = zeros(6, 1);

    for i = 1:6
        region = regions{i};
        xmin = round(region(1) * w);
        xmax = round(region(2) * w);
        ymin = round(region(3) * h);
        ymax = round(region(4) * h);
        cropped_img = img(ymin+1:ymax, xmin+1:xmax, :);

        fig = figure('Name', sprintf('Sample %d: %s', sample_id, labels{i}));
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
        imshow(cropped_img, 'InitialMagnification', 'fit');
        axis on;
        axis image;
        title(sprintf('Sample %d: %s', sample_id, labels{i}), 'FontSize', 14);

        [xi_crop, yi_crop] = ginput(1);
        close(fig);

        x(i) = xi_crop + xmin;
        y(i) = yi_crop + ymin;
    end

    % Update the correct 6 rows in T using Sample # and Label
    for j = 1:6
        row_idx = (T.Sample == sample_id) & strcmp(T.Label, labels{j});
        if ~isempty(row_idx)
            T.X(row_idx) = x(j);
            T.Y(row_idx) = y(j);
            T.PatientID(row_idx) = repmat(string(patient_id), sum(row_idx), 1);
        end
    end

    % Optional: show full image
    show_full = input('Display full image with all landmarks? (Y/N): ', 's');
    if strcmpi(show_full, 'Y')
        fig = figure('Name', sprintf('Sample %d: Annotated Full Image', sample_id));
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
        imshow(img, 'InitialMagnification', 'fit');
        hold on;

        for i = 1:6
            plot(x(i), y(i), 'o', 'MarkerSize', 6, ...
                 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
            if i <= 3
                text(x(i) - 10, y(i), labels{i}, 'Color', colors{i}, ...
                     'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
            else
                text(x(i) + 5, y(i), labels{i}, 'Color', colors{i}, ...
                     'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
            end
        end

        title(sprintf('Sample %d: Annotated Full Image', sample_id), 'FontSize', 14);
        hold off;

        cont = input('Continue to next sample? (Y/N): ', 's');
        if isvalid(fig)
            close(fig);
        end
        if strcmpi(cont, 'N')
            break;
        end
    end
end

writetable(T, filename);
disp('All annotations saved to outputs.xlsx');

