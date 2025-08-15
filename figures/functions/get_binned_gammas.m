function binned_gammas_fent = get_binned_gammas(gammas_per_sess,numBins)
            
% Get the number of rows and columns in the matrix
[numtrials,numCol] = size(gammas_per_sess);
            
% Calculate the number of rows per part
rowsPerPart = floor(numtrials / numBins);
extraRows = mod(numtrials, numBins);  % Remainder rows that need to be distributed
            
% Create a cell array to store the resulting parts
matrices = cell(1, numBins);
            
% Index for splitting the matrix
startIdx = 1;
for k = 1:numBins
    % If there are extra rows to distribute, give this part an extra row
    if k <= extraRows
        endIdx = startIdx + rowsPerPart;  % One extra row for this part
    else
        endIdx = startIdx + rowsPerPart - 1;  % Normal number of rows
    end
    
    % Store the split matrix part
    matrices{k} = gammas_per_sess(startIdx:endIdx,:);

    % Update the starting index for the next part
    startIdx = endIdx + 1;

    % average content in matrices
    binned_gammas_fent(k,:) = mean(matrices{k});  
end