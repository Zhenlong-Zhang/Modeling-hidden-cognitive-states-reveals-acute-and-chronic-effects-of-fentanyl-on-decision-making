function build_and_export_cleaned_data()
% build_and_export_cleaned_data
% -----------------------------
% read all.mat files under this foder "zhenlong"
% create a sub-folder name based on the name of .mat. 
% put the data in the sub-folder
% data contains 1. chioce(1,2) 2. tran(0,1) 3. reward(0,1) 4. new session(0,1) 

    folderName = 'zhenlong';
    matFiles = dir(fullfile(folderName, '*.mat'));
    
    % save to
    outDir = 'data-cleaned';
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % read all .mat
    for i = 1:length(matFiles)
        fullName = fullfile(folderName, matFiles(i).name);
        fprintf('Loading file: %s\n', fullName);
        load(fullName, 'res'); 
        
        if ~isfield(res, 'concat_data')
            warning('File %s does not contain res.concat_data, skipping.', fullName);
            continue;
        end
        
        cData = res.concat_data;
        ratNames = fieldnames(cData);
        
        % based on its name.mat create sub-folder
        [~, fileBaseName, ~] = fileparts(matFiles(i).name);
        fileOutDir = fullfile(outDir, fileBaseName);
        if ~exist(fileOutDir, 'dir')
            mkdir(fileOutDir);
        end
        
        % deal with each rat
        for r = 1:length(ratNames)
            rn = ratNames{r};  %  'rat3'
            
            sessions = {};
            
            
            sessionArray = cData.(rn);
            if iscell(sessionArray)
                for s = 1:length(sessionArray)
                    sessions{end+1} = sessionArray{s};
                end
            elseif isstruct(sessionArray)
                for s = 1:length(sessionArray)
                    sessions{end+1} = sessionArray(s);
                end
            else
                error('Unexpected type for cData.(%s): %s', rn, class(sessionArray));
            end
            
         
            try
                [choice, trans, reward, new_sess] = concat_sessions(sessions);
            catch ME
                warning('Error processing %s in file %s: %s', rn, matFiles(i).name, ME.message);
                continue;
            end
            
       
            T = table(choice, trans, reward, new_sess, 'VariableNames', {'Choice','Trans','Reward','NewSess'});
            
          
            outFileName = sprintf('%s_data.csv', rn);
            outPath = fullfile(fileOutDir, outFileName);
            writetable(T, outPath);
            fprintf('Exported data for %s to %s\n', rn, outPath);
        end
    end
    
    fprintf('All data exported to folder "%s".\n', outDir);
end

% --- : concat_sessions ---
function [choice, trans, reward, new_sess] = concat_sessions(sections)
% concat_sessions
% ----------------

%    Column 1  Choice，
%    Column 2  Trans，
%    Column 4  Reward。


    choice = [];
    trans = [];
    reward = [];
    new_sess = [];
    
    for s = 1:length(sections)
        sessionData = sections{s};
       
        if ~isstruct(sessionData)
            if iscell(sessionData) && ~isempty(sessionData) && isstruct(sessionData{1})
                sessionData = sessionData{1};
            else
                error('Session %d is of unsupported type: %s', s, class(sessionData));
            end
        end
        
        if ~isfield(sessionData, 'CTSR')
            error('Session %d does not contain field CTSR.', s);
        end
        
        CTSR = sessionData.CTSR;
        
        choice = [choice; CTSR(:,1)];
        trans  = [trans;  CTSR(:,2)];
        reward = [reward; CTSR(:,4)];
        
        nrows = size(CTSR, 1);
        new_sess = [new_sess; true; false(nrows-1, 1)];
    end
end
