% LL as a function of number of states

function ll = get_ll(max_states,struc)

for i = 1:max_states
    
    max_states_string = num2str(i);
    field_name{i} = ['nStates',max_states_string];
    
    ll(i) = struc.(field_name{i}).logLik;
    
end

end




