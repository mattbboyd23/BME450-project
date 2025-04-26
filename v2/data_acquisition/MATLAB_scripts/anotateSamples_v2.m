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

zoom_window = 101;  % Size of zoom-in square (must be odd)

for sample_id = start_id:end_id
    files = dir(fullfile(folder, sprintf('sample%d-*-resized.jpg', sample_id)));

    if isempty(files)
        warning('No resized file found for sample %d. Skipping...', sample_id);
        continue;
    end

    fname = files(1).name;

    % Extract patient ID from filename
    tokens = regexp(fname, 'sample(\d+)-([\d]{7})-resized\.jpg', 'tokens');
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

        % Coarse click figure
        fig = figure('Name', sprintf('Sample %d: %s (Coarse)', sample_id, labels{i}));
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
        imshow(cropped_img, 'InitialMagnification', 'fit');
        axis on; axis image;
        title(sprintf('Sample %d: %s (Coarse)', sample_id, labels{i}), 'FontSize', 14);
        [xi_crop, yi_crop] = ginput(1);
        close(fig);

        xi_full = xi_crop + xmin;
        yi_full = yi_crop + ymin;

        % Center of zoom window
        x_center = round(xi_full);
        y_center = round(yi_full);
        half_win = floor(zoom_window / 2);

        % Initial bounds
        x1 = max(1, x_center - half_win);
        y1 = max(1, y_center - half_win);
        x2 = min(w, x1 + zoom_window - 1);
        y2 = min(h, y1 + zoom_window - 1);

        % Adjust if near edge
        if x2 - x1 + 1 < zoom_window
            x1 = max(1, x2 - zoom_window + 1);
        end
        if y2 - y1 + 1 < zoom_window
            y1 = max(1, y2 - zoom_window + 1);
        end

        zoom_patch = img(y1:y2, x1:x2, :);

        % Refined click on zoomed-in patch
        fig = figure('Name', sprintf('Sample %d: %s (Zoomed)', sample_id, labels{i}));
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
        imshow(zoom_patch, 'InitialMagnification', 'fit');
        axis on; axis image;
        title(sprintf('Sample %d: %s (Zoomed)', sample_id, labels{i}), 'FontSize', 14);
        hold on;
        % Green point
        plot(half_win+1, half_win+1, 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
        hold off;
        [xi_zoom, yi_zoom] = ginput(1);
        close(fig);

        % Final refined coordinate in full image
        x(i) = x1 + xi_zoom - 1;
        y(i) = y1 + yi_zoom - 1;
    end

    % Update outputs.xlsx
    for j = 1:6
        row_idx = (T.Sample == sample_id) & strcmp(T.Label, labels{j});
        if ~isempty(row_idx)
            T.X(row_idx) = x(j);
            T.Y(row_idx) = y(j);
            T.PatientID(row_idx) = repmat(string(patient_id), sum(row_idx), 1);
        end
    end

    % Always display and save annotated full image
    fig = figure('Name', sprintf('Sample %d: Annotated Full Image', sample_id));
    set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    imshow(img, 'InitialMagnification', 'fit');
    hold on;
    for i = 1:6
        plot(x(i), y(i), 'o', 'MarkerSize', 6, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
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

    % Save and prompt
    saveas(fig, sprintf('sample%d-annotated.jpg', sample_id));

    cont = input('Continue to next sample? (Y/N): ', 's');
    if isvalid(fig), close(fig); end
    if strcmpi(cont, 'N')
        break;
    end
end

writetable(T, filename);
disp('All annotations saved and figures exported.');