from deep_translator import GoogleTranslator
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk

def calculate_metrics(reference, candidate):
    """
    Calculate BLEU and ROUGE scores for translation quality.
    """
    try:
        # BLEU score calculation
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
        
        # ROUGE score calculation
        rouge = Rouge()
        rouge_scores = rouge.get_scores(candidate, reference)
        
        return {
            'bleu': bleu_score,
            'rouge-1': rouge_scores[0]['rouge-1']['f'],
            'rouge-2': rouge_scores[0]['rouge-2']['f'],
            'rouge-l': rouge_scores[0]['rouge-l']['f']
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def translate_text(text, target_lang='en'):
    """
    Translates text to the specified target language using GoogleTranslator.
    """
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return None

def main():
    # List of target languages (10 languages)
    target_languages = {
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese'
    }
    
    # Get input text from user
    input_text = input("Enter the text to translate: ")
    print("\nTranslating to 10 languages and calculating metrics...\n")
    
    # Translate to each language and calculate metrics
    for lang_code, lang_name in target_languages.items():
        translation = translate_text(input_text, lang_code)
        if translation:
            print(f"\n{lang_name} translation:")
            print(translation)
            
            # Calculate back translation for metrics
            back_translation = translate_text(translation, 'en')
            if back_translation:
                metrics = calculate_metrics(input_text, back_translation)
                if metrics:
                    print(f"Metrics for {lang_name}:")
                    print(f"BLEU Score: {metrics['bleu']:.4f}")
                    print(f"ROUGE-1 F1: {metrics['rouge-1']:.4f}")
                    print(f"ROUGE-2 F1: {metrics['rouge-2']:.4f}")
                    print(f"ROUGE-L F1: {metrics['rouge-l']:.4f}")
                    print("-" * 50)

if __name__ == "__main__":
    nltk.download('punkt')  # Download required NLTK data
    main()