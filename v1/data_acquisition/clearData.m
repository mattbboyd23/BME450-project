filename = 'outputs.xlsx';

% Read the existing Excel file
if ~isfile(filename)
    error('File "%s" not found.', filename);
end

T = readtable(filename);

% Prompt user
user_input = input('Enter sample number to clear (e.g., 4), or type "all": ', 's');

if strcmpi(user_input, 'all')
    % Confirm irreversible action
    confirm = input('Are you sure you want to clear ALL data? (Y/N): ', 's');
    
    if strcmpi(confirm, 'Y')
        T.X(:) = NaN;
        T.Y(:) = NaN;
        disp('All coordinate data cleared.');
    else
        disp('Operation canceled. No data was modified.');
        return;
    end
else
    % Try to convert input to a sample ID number
    sample_id = str2double(user_input);
    if isnan(sample_id)
        error('Invalid input. Please enter a number or "all".');
    end

    sample_name = sprintf('sample%d.png', sample_id);
    rows_to_clear = strcmp(T.Image, sample_name);

    if any(rows_to_clear)
        T.X(rows_to_clear) = NaN;
        T.Y(rows_to_clear) = NaN;
        fprintf('Cleared data for sample %d (%s).\n', sample_id, sample_name);
    else
        warning('Sample %d not found in the data.', sample_id);
    end
end

% Save the updated table
writetable(T, filename);
disp('Updated outputs.xlsx saved.');