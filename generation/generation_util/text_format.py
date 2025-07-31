
def replace_special_quotes_to_straight_quotes(input_prompt: str) -> str:
    """Replace special quotes to straight quotes"""
    if not input_prompt:
        return input_prompt
        
    # Replace various types of curly quotes with straight quotes
    replacements = {
        '“': '"',  # Left double quotation mark
        '”': '"',  # Right double quotation mark         
    }
    
    result = input_prompt
    for special_quote, straight_quote in replacements.items():
        if special_quote in result:
            result = result.replace(special_quote, straight_quote)
    return result
