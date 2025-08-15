% betas as a function of number of states

function betas = get_betas(max_states,struc)

for i = 1:max_states
    
    max_states_string = num2str(i);
    field_name{i} = ['nStates',max_states_string];
    
    betas{i} = struc.(field_name{i}).freeBeta';
    
end

end
