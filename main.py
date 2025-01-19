from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk

app = Flask(__name__)

# Download required NLTK data at startup
nltk.download('punkt')

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
        return {'error': str(e)}

def translate_text(text, target_lang='en', max_retries=3, timeout=5):
    """
    Translates text to the specified target language using GoogleTranslator.
    Added retry mechanism and timeout.
    """
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return None
            continue

# List of supported languages
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'te': 'Telugu',
    'bn': 'Bengali',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia'
}

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Translation API',
        'supported_languages': SUPPORTED_LANGUAGES,
        'endpoints': {
            '/translate': 'POST request with {"text": "your text", "target_lang": "language_code"}',
            '/translate_all': 'POST request with {"text": "your text"}'
        }
    })

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    
    if not data or 'text' not in data or 'target_lang' not in data:
        return jsonify({'error': 'Please provide both text and target_lang'}), 400
    
    text = data['text']
    target_lang = data['target_lang']
    
    if target_lang not in SUPPORTED_LANGUAGES:
        return jsonify({'error': 'Unsupported target language'}), 400
    
    translation = translate_text(text, target_lang)
    if not translation:
        return jsonify({'error': 'Translation failed'}), 500
    
    # Get back translation for metrics
    back_translation = translate_text(translation, 'en')
    metrics = calculate_metrics(text, back_translation) if back_translation else None
    
    return jsonify({
        'original_text': text,
        'translated_text': translation,
        'language': SUPPORTED_LANGUAGES[target_lang],
        'metrics': metrics
    })

@app.route('/translate_all', methods=['POST'])
def translate_all():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text to translate'}), 400
    
    text = data['text']
    results = {}
    
    # Split languages into smaller batches
    BATCH_SIZE = 3
    language_batches = [list(SUPPORTED_LANGUAGES.items())[i:i + BATCH_SIZE] 
                       for i in range(0, len(SUPPORTED_LANGUAGES), BATCH_SIZE)]
    
    for batch in language_batches:
        batch_results = {}
        for lang_code, lang_name in batch:
            translation = translate_text(text, lang_code)
            if translation:
                # Only do back translation if forward translation succeeded
                back_translation = translate_text(translation, 'en')
                metrics = calculate_metrics(text, back_translation) if back_translation else None
                
                batch_results[lang_code] = {
                    'language': lang_name,
                    'translated_text': translation,
                    'metrics': metrics
                }
        results.update(batch_results)
    
    return jsonify({
        'original_text': text,
        'translations': results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
