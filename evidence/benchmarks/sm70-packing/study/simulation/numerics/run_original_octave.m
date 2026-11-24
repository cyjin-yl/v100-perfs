% Run the authors' unmodified V100 MATLAB model in GNU Octave.
% The input vectors below are already exactly rounded to FP16/FP32.
% CPFloat is optional in upstream GEMM.m and is not needed for these inputs.
base = fileparts(mfilename('fullpath'));
repo = getenv('V100_MODEL_UPSTREAM');
if isempty(repo)
    repo = fullfile(base,'MATLAB-tensor-core');
    if ~exist(fullfile(repo,'models','V100TC.m'),'file')
        repo = fullfile(base,'upstream-snapshot');
    end
end
addpath(fullfile(repo,'models'));
addpath(fullfile(repo,'models','tools'));
addpath(fullfile(repo,'model_validation'));
data = fullfile(repo,'model_validation','V100','fp16');
A = readHexFloatFile(fullfile(data,'a_V100_fp16.txt'));
B = readHexFloatFile(fullfile(data,'b_V100_fp16.txt'));
% Same reshape as the authors' Validate_TC_models.m: their reader returns
% a flat column on Octave because typecast preserves hex2dec's column shape.
A = reshape(A,4,numel(A)/4).';
B = reshape(B,4,numel(B)/4).';
C = readIeeeFloatsFromFile(fullfile(data,'c_V100_fp32.txt'));
D = readIeeeFloatsFromFile(fullfile(data,'d_V100_fp32.txt'));
n = numel(C);
fixture_errors = 0;
t = tic();
for index=1:n
    a = double(A(index,:));
    b = double(B(index,:)).';
    c = double(C(index));
    % Capture the upstream optional-CPFloat notice, without modifying source.
    evalc("got = V100TC(1,a,b,1,c,'fp32');");
    fixture_errors = fixture_errors + (typecast(single(got),'uint32') ~= typecast(D(index),'uint32'));
end
fixture_seconds = toc(t);
cases = jsondecode(fileread(fullfile(base,'original_model_cases.json')));
case_errors = 0;
case_results = cell(numel(cases),1);
t = tic();
for index=1:numel(cases)
    if iscell(cases), item=cases{index}; else, item=cases(index); end
    a = double(item.a(:)).';
    b = double(item.b(:));
    c = double(item.c);
    evalc("got = V100TC(1,a,b,1,c,'fp32');");
    raw = typecast(single(got),'uint32');
    bad = raw ~= uint32(hex2dec(item.expected_hex));
    case_errors = case_errors + bad;
    case_results{index} = struct('label',item.label,'result',got,'result_hex',dec2hex(raw,8),'mismatch',logical(bad));
end
case_seconds = toc(t);
results = struct('runtime',version(),'original_upstream_code_executed',true,...
    'upstream_commit','6fd0520850164bdd1c37f901954e27ec63eada56',...
    'fixture_count',n,'fixture_bit_mismatches',fixture_errors,...
    'fixture_seconds',fixture_seconds,'packing_and_control_case_count',numel(cases),...
    'packing_and_control_bit_mismatches',case_errors,'case_seconds',case_seconds,...
    'cases',{case_results});
fid=fopen(fullfile(base,'original_octave_results.json'),'w');
fprintf(fid,'%s\n',jsonencode(results));
fclose(fid);
fprintf('Original V100 model: %d/%d hardware fixtures match; %d/%d packing/control cases match Python port.\n',...
    n-fixture_errors,n,numel(cases)-case_errors,numel(cases));
assert(fixture_errors==0);
assert(case_errors==0);
