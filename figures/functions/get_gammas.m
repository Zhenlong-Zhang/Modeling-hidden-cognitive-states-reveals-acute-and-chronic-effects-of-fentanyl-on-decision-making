% gammas as a function of number of states

function trans_probs = get_gammas(max_states,struc)

for i = 1:max_states
    
    max_states_string = num2str(i);
    field_name{i} = ['nStates',max_states_string];
    
    trans_probs{i} = struc.(field_name{i}).gamma;
    
end

end
