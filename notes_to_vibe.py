import numpy as np

# Define vibe dimensions
VIBE_DIMENSIONS = ['formality', 'season', 'frequency', 'gender', 'time', 'longevity']

# Create a dictionary to translate vibes to notes (TO CHANGE)
NOTES_VIBE_DICT = {
    'leather':      {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
    'lemon':        {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
    'vanilla':      {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
    'oud':          {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
    'bergamot':     {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
    'sandalwood':   {'formality': 0.0, 'season': 0.0, 'frequency': 0.0, 'gender': 0.0, 'time': 0.0, 'longevity': 0.0},
}

def calculate_vibe_vector(notes_list):
    '''
    Takes a list of string notes, looks them up in the dictionary, 
    and averages their scores to output a baseline vibe vector.
    '''
    vectors = []
    
    for note in notes_list:
        note_clean = str(note).lower().strip()
        if note_clean in NOTES_VIBE_DICT:
            vec = [NOTES_VIBE_DICT[note_clean][dim] for dim in VIBE_DIMENSIONS]
            vectors.append(vec)
            
    # Default if no notes match
    if not vectors:
        return np.zeros(len(VIBE_DIMENSIONS)) 
        
    # Average the vectors to get the overall fragrance profile
    return np.mean(vectors, axis=0)

# --- Test the Mapping ---
test_perfume_notes = ['lemon', 'bergamot', 'sandalwood']
vibe_vector = calculate_vibe_vector(test_perfume_notes)

print(f'Notes: {test_perfume_notes}')
print(f'Calculated Vibe Vector {VIBE_DIMENSIONS}:')
print(np.round(vibe_vector, 2)) 