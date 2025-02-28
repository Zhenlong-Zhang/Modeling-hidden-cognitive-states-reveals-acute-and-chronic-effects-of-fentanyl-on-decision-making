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
        
        % 对当前文件中每只大鼠的数据进行处理
        for r = 1:length(ratNames)
            rn = ratNames{r};  % 例如 'rat3'
            % 初始化 cell 数组存放该 rat 的所有 session 数据
            sessions = {};
            
            % 假设 cData.(rn) 为一个 struct 数组或 cell 数组，每个元素代表一个 session
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
            
            % 调用 concat_sessions 将多个 session 拼接为最终数据
            try
                [choice, trans, reward, new_sess] = concat_sessions(sessions);
            catch ME
                warning('Error processing %s in file %s: %s', rn, matFiles(i).name, ME.message);
                continue;
            end
            
            % 生成 table 数据
            T = table(choice, trans, reward, new_sess, 'VariableNames', {'Choice','Trans','Reward','NewSess'});
            
            % 写出 CSV 文件，例如 "rat3_data.csv"
            outFileName = sprintf('%s_data.csv', rn);
            outPath = fullfile(fileOutDir, outFileName);
            writetable(T, outPath);
            fprintf('Exported data for %s to %s\n', rn, outPath);
        end
    end
    
    fprintf('All data exported to folder "%s".\n', outDir);
end

% --- 内部函数: concat_sessions ---
function [choice, trans, reward, new_sess] = concat_sessions(sections)
% concat_sessions
% ----------------
% 将多个 session 的数据合并为长向量。
% 每个 session 的数据存放在结构体中，必须包含字段 CTSR，其中：
%    Column 1 为 Choice，
%    Column 2 为 Trans，
%    Column 4 为 Reward。
% 同时，每个 session 的首个 trial 标记 new_sess 为 true，其余为 false.
%
% Input:
%   sections - cell 数组，每个元素为一个 session 的数据
% Output:
%   choice, trans, reward, new_sess - 合并后的列向量

    choice = [];
    trans = [];
    reward = [];
    new_sess = [];
    
    for s = 1:length(sections)
        sessionData = sections{s};
        % 检查 sessionData 是否为结构体，不是则尝试取其第一个元素
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
        % 按照要求：第1列为 Choice, 第2列为 Trans, 第4列为 Reward
        choice = [choice; CTSR(:,1)];
        trans  = [trans;  CTSR(:,2)];
        reward = [reward; CTSR(:,4)];
        
        nrows = size(CTSR, 1);
        new_sess = [new_sess; true; false(nrows-1, 1)];
    end
end
