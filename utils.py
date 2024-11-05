def find_buggy_token_positions(buggy_code, fixed_code):
    buggy_tokens = buggy_code.split()
    fixed_tokens = fixed_code.split()
    
    start_idx, end_idx = None, None
    
    min_length = min(len(buggy_tokens), len(fixed_tokens))
    for i in range(min_length):
        if buggy_tokens[i] != fixed_tokens[i]:
            start_idx = i
            break
    
    for i in range(1, min_length + 1):
        if buggy_tokens[-i] != fixed_tokens[-i]:
            end_idx = len(buggy_tokens) - i
            break
    
    # Ass the below case:
    # static java.lang.String METHOD_1 ( ) { return VAR_1 ; } 
    # public static java.lang.String METHOD_1 ( ) { return VAR_1 ; } 
    if start_idx is not None and end_idx is None:
        end_idx = start_idx

    return start_idx, end_idx