filename = 'outputs.xlsx';

landmarks = {'RH', 'RK', 'RA', 'LH', 'LK', 'LA'};
num_samples = 100;
rows_per_sample = length(landmarks);
total_rows = num_samples * rows_per_sample;

% Preallocate
PatientID = strings(total_rows, 1);
Sample = zeros(total_rows, 1);
Label = strings(total_rows, 1);
X = nan(total_rows, 1);
Y = nan(total_rows, 1);

row = 1;
for sample_id = 1:num_samples
    for j = 1:rows_per_sample
        PatientID(row) = "";        % To be filled during annotation
        Sample(row) = sample_id;    % Just the number (e.g., 1, 2, 3...)
        Label(row) = landmarks{j};  % RH, RK, etc.
        row = row + 1;
    end
end

T = table(PatientID, Sample, Label, X, Y);
writetable(T, filename);
disp('Initialized outputs.xlsx with 100 samples and 600 landmark rows.');
