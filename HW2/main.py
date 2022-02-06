from preprocessing import  string_to_lower, load_text, splitting_tokens,lists_to_tokens






def main():
    text = load_text('wiki.train.txt')
    text = string_to_lower(text)
    text =  splitting_tokens(text)
    text = lists_to_tokens(text)



    print(text[240])









if __name__ =='__main__':
    main()
