import numpy as np

# Define vibe dimensions
VIBE_DIMENSIONS = ['formality', 'season', 'frequency', 'gender', 'time', 'longevity']

# Dimensions:
#   formality  : 0=very casual (citrus, aquatic)  → 1=very formal (oud, incense, resinous)
#   season     : 0=summer/hot weather (citrus, aquatic, light florals) → 1=winter/cold (oud, vanilla, woods, spices)
#   frequency  : 0=everyday wear (fresh, clean, light) → 1=special occasion (heavy orientals, intense musks)
#   gender     : 0=traditionally feminine (rose, peony, fruity florals) → 1=traditionally masculine (leather, oud, vetiver)
#               0.5=unisex
#   time       : 0=morning/daytime (citrus, green, clean) → 1=evening/nighttime (oud, amber, heavy musk, incense)
#   longevity  : 0=fleeting/top note (citrus, light aldehydes) → 1=long-lasting/base note (oud, amber, vanilla, musks, woods)

NOTES_VIBE_DICT = {
    # ── High-frequency corpus notes ──────────────────────────────────────────
    'musk':                  {'formality': 0.45, 'season': 0.50, 'frequency': 0.45, 'gender': 0.50, 'time': 0.60, 'longevity': 0.85},
    'bergamot':              {'formality': 0.30, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},
    'amber':                 {'formality': 0.65, 'season': 0.80, 'frequency': 0.60, 'gender': 0.55, 'time': 0.80, 'longevity': 0.90},
    'patchouli':             {'formality': 0.55, 'season': 0.75, 'frequency': 0.55, 'gender': 0.55, 'time': 0.75, 'longevity': 0.90},
    'vanilla':               {'formality': 0.40, 'season': 0.85, 'frequency': 0.40, 'gender': 0.30, 'time': 0.70, 'longevity': 0.88},
    'sandalwood':            {'formality': 0.60, 'season': 0.70, 'frequency': 0.50, 'gender': 0.55, 'time': 0.65, 'longevity': 0.85},
    'jasmine':               {'formality': 0.55, 'season': 0.45, 'frequency': 0.50, 'gender': 0.20, 'time': 0.60, 'longevity': 0.55},
    'rose':                  {'formality': 0.60, 'season': 0.40, 'frequency': 0.45, 'gender': 0.15, 'time': 0.50, 'longevity': 0.50},
    'cedarwood':             {'formality': 0.55, 'season': 0.65, 'frequency': 0.45, 'gender': 0.65, 'time': 0.55, 'longevity': 0.80},
    'lemon':                 {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.45, 'time': 0.05, 'longevity': 0.10},
    'vetiver':               {'formality': 0.65, 'season': 0.65, 'frequency': 0.55, 'gender': 0.75, 'time': 0.65, 'longevity': 0.85},
    'tonka bean':            {'formality': 0.50, 'season': 0.80, 'frequency': 0.50, 'gender': 0.45, 'time': 0.70, 'longevity': 0.85},
    'mandarin orange':       {'formality': 0.20, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},
    'lavender':              {'formality': 0.35, 'season': 0.40, 'frequency': 0.20, 'gender': 0.60, 'time': 0.35, 'longevity': 0.35},
    'orange blossom':        {'formality': 0.50, 'season': 0.30, 'frequency': 0.35, 'gender': 0.25, 'time': 0.45, 'longevity': 0.45},
    'grapefruit':            {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.50, 'time': 0.05, 'longevity': 0.10},
    'geranium':              {'formality': 0.40, 'season': 0.40, 'frequency': 0.35, 'gender': 0.45, 'time': 0.40, 'longevity': 0.45},
    'pink pepper':           {'formality': 0.45, 'season': 0.50, 'frequency': 0.40, 'gender': 0.50, 'time': 0.45, 'longevity': 0.30},
    'lily of the valley':    {'formality': 0.50, 'season': 0.25, 'frequency': 0.35, 'gender': 0.10, 'time': 0.30, 'longevity': 0.35},
    'cardamom':              {'formality': 0.60, 'season': 0.65, 'frequency': 0.55, 'gender': 0.60, 'time': 0.65, 'longevity': 0.40},
    'cedar':                 {'formality': 0.55, 'season': 0.60, 'frequency': 0.45, 'gender': 0.65, 'time': 0.55, 'longevity': 0.80},
    'oakmoss':               {'formality': 0.65, 'season': 0.65, 'frequency': 0.60, 'gender': 0.60, 'time': 0.65, 'longevity': 0.80},
    'iris':                  {'formality': 0.70, 'season': 0.50, 'frequency': 0.60, 'gender': 0.30, 'time': 0.55, 'longevity': 0.60},
    'violet':                {'formality': 0.50, 'season': 0.35, 'frequency': 0.40, 'gender': 0.20, 'time': 0.45, 'longevity': 0.40},
    'rosemary':              {'formality': 0.30, 'season': 0.40, 'frequency': 0.20, 'gender': 0.55, 'time': 0.25, 'longevity': 0.25},
    'leather':               {'formality': 0.80, 'season': 0.70, 'frequency': 0.75, 'gender': 0.85, 'time': 0.80, 'longevity': 0.85},
    'ylang-ylang':           {'formality': 0.55, 'season': 0.50, 'frequency': 0.50, 'gender': 0.20, 'time': 0.55, 'longevity': 0.55},
    'peach':                 {'formality': 0.25, 'season': 0.20, 'frequency': 0.25, 'gender': 0.20, 'time': 0.30, 'longevity': 0.25},
    'orange':                {'formality': 0.20, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},
    'ambergris':             {'formality': 0.70, 'season': 0.70, 'frequency': 0.65, 'gender': 0.55, 'time': 0.75, 'longevity': 0.90},
    'blackcurrant':          {'formality': 0.25, 'season': 0.30, 'frequency': 0.25, 'gender': 0.25, 'time': 0.30, 'longevity': 0.20},
    'white musk':            {'formality': 0.30, 'season': 0.35, 'frequency': 0.25, 'gender': 0.35, 'time': 0.45, 'longevity': 0.80},
    'saffron':               {'formality': 0.80, 'season': 0.80, 'frequency': 0.75, 'gender': 0.60, 'time': 0.80, 'longevity': 0.55},
    'neroli':                {'formality': 0.50, 'season': 0.20, 'frequency': 0.30, 'gender': 0.35, 'time': 0.25, 'longevity': 0.25},
    'labdanum':              {'formality': 0.70, 'season': 0.80, 'frequency': 0.70, 'gender': 0.60, 'time': 0.80, 'longevity': 0.88},
    'frankincense':          {'formality': 0.85, 'season': 0.80, 'frequency': 0.80, 'gender': 0.60, 'time': 0.85, 'longevity': 0.80},
    'oud':                   {'formality': 0.95, 'season': 0.90, 'frequency': 0.90, 'gender': 0.75, 'time': 0.95, 'longevity': 0.98},
    'freesia':               {'formality': 0.40, 'season': 0.25, 'frequency': 0.30, 'gender': 0.15, 'time': 0.30, 'longevity': 0.30},
    'ginger':                {'formality': 0.40, 'season': 0.55, 'frequency': 0.35, 'gender': 0.55, 'time': 0.45, 'longevity': 0.30},
    'apple':                 {'formality': 0.15, 'season': 0.20, 'frequency': 0.15, 'gender': 0.35, 'time': 0.15, 'longevity': 0.15},
    'black pepper':          {'formality': 0.55, 'season': 0.55, 'frequency': 0.50, 'gender': 0.65, 'time': 0.55, 'longevity': 0.30},
    'pineapple':             {'formality': 0.10, 'season': 0.10, 'frequency': 0.10, 'gender': 0.50, 'time': 0.10, 'longevity': 0.15},
    'cinnamon':              {'formality': 0.55, 'season': 0.80, 'frequency': 0.55, 'gender': 0.55, 'time': 0.65, 'longevity': 0.55},
    'moss':                  {'formality': 0.55, 'season': 0.60, 'frequency': 0.55, 'gender': 0.60, 'time': 0.60, 'longevity': 0.75},
    'violet leaf':           {'formality': 0.40, 'season': 0.35, 'frequency': 0.35, 'gender': 0.40, 'time': 0.35, 'longevity': 0.35},
    'peony':                 {'formality': 0.50, 'season': 0.30, 'frequency': 0.35, 'gender': 0.05, 'time': 0.35, 'longevity': 0.35},
    'raspberry':             {'formality': 0.20, 'season': 0.20, 'frequency': 0.20, 'gender': 0.15, 'time': 0.25, 'longevity': 0.20},
    'benzoin':               {'formality': 0.60, 'season': 0.80, 'frequency': 0.60, 'gender': 0.50, 'time': 0.75, 'longevity': 0.80},
    'marine notes':          {'formality': 0.10, 'season': 0.05, 'frequency': 0.10, 'gender': 0.60, 'time': 0.10, 'longevity': 0.15},
    'tuberose':              {'formality': 0.70, 'season': 0.45, 'frequency': 0.65, 'gender': 0.10, 'time': 0.70, 'longevity': 0.65},
    'mint':                  {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.50, 'time': 0.05, 'longevity': 0.10},
    'magnolia':              {'formality': 0.50, 'season': 0.30, 'frequency': 0.40, 'gender': 0.15, 'time': 0.40, 'longevity': 0.40},
    'pear':                  {'formality': 0.20, 'season': 0.15, 'frequency': 0.15, 'gender': 0.25, 'time': 0.15, 'longevity': 0.15},
    'nutmeg':                {'formality': 0.50, 'season': 0.70, 'frequency': 0.50, 'gender': 0.60, 'time': 0.55, 'longevity': 0.45},
    'clary sage':            {'formality': 0.40, 'season': 0.45, 'frequency': 0.35, 'gender': 0.55, 'time': 0.40, 'longevity': 0.40},
    'lily':                  {'formality': 0.50, 'season': 0.30, 'frequency': 0.40, 'gender': 0.10, 'time': 0.40, 'longevity': 0.40},
    'pepper':                {'formality': 0.50, 'season': 0.55, 'frequency': 0.45, 'gender': 0.60, 'time': 0.50, 'longevity': 0.30},
    'sage':                  {'formality': 0.40, 'season': 0.50, 'frequency': 0.35, 'gender': 0.60, 'time': 0.40, 'longevity': 0.35},
    'coriander':             {'formality': 0.40, 'season': 0.45, 'frequency': 0.35, 'gender': 0.55, 'time': 0.40, 'longevity': 0.25},
    'heliotrope':            {'formality': 0.50, 'season': 0.50, 'frequency': 0.45, 'gender': 0.25, 'time': 0.50, 'longevity': 0.55},
    'hyacinth':              {'formality': 0.55, 'season': 0.30, 'frequency': 0.45, 'gender': 0.20, 'time': 0.40, 'longevity': 0.40},
    'aquatic notes':         {'formality': 0.10, 'season': 0.05, 'frequency': 0.10, 'gender': 0.55, 'time': 0.05, 'longevity': 0.10},
    'galbanum':              {'formality': 0.55, 'season': 0.50, 'frequency': 0.50, 'gender': 0.50, 'time': 0.50, 'longevity': 0.55},
    'damask rose':           {'formality': 0.70, 'season': 0.45, 'frequency': 0.60, 'gender': 0.10, 'time': 0.60, 'longevity': 0.55},
    'cyclamen':              {'formality': 0.35, 'season': 0.25, 'frequency': 0.30, 'gender': 0.20, 'time': 0.30, 'longevity': 0.30},
    'caramel':               {'formality': 0.25, 'season': 0.75, 'frequency': 0.35, 'gender': 0.25, 'time': 0.60, 'longevity': 0.55},
    'petitgrain':            {'formality': 0.30, 'season': 0.20, 'frequency': 0.20, 'gender': 0.50, 'time': 0.20, 'longevity': 0.25},
    'honey':                 {'formality': 0.45, 'season': 0.60, 'frequency': 0.45, 'gender': 0.35, 'time': 0.55, 'longevity': 0.60},
    'gaiac wood':            {'formality': 0.65, 'season': 0.70, 'frequency': 0.60, 'gender': 0.65, 'time': 0.65, 'longevity': 0.80},
    'osmanthus':             {'formality': 0.60, 'season': 0.45, 'frequency': 0.55, 'gender': 0.30, 'time': 0.55, 'longevity': 0.55},
    'basil':                 {'formality': 0.25, 'season': 0.30, 'frequency': 0.20, 'gender': 0.55, 'time': 0.20, 'longevity': 0.20},
    'plum':                  {'formality': 0.45, 'season': 0.55, 'frequency': 0.45, 'gender': 0.25, 'time': 0.55, 'longevity': 0.35},
    'gardenia':              {'formality': 0.60, 'season': 0.35, 'frequency': 0.55, 'gender': 0.10, 'time': 0.55, 'longevity': 0.50},
    'myrrh':                 {'formality': 0.80, 'season': 0.80, 'frequency': 0.75, 'gender': 0.60, 'time': 0.85, 'longevity': 0.85},
    'styrax':                {'formality': 0.65, 'season': 0.75, 'frequency': 0.65, 'gender': 0.55, 'time': 0.75, 'longevity': 0.80},
    'clove':                 {'formality': 0.65, 'season': 0.80, 'frequency': 0.65, 'gender': 0.60, 'time': 0.70, 'longevity': 0.60},
    'blackberry':            {'formality': 0.20, 'season': 0.25, 'frequency': 0.20, 'gender': 0.30, 'time': 0.25, 'longevity': 0.20},
    'atlas cedar':           {'formality': 0.55, 'season': 0.60, 'frequency': 0.45, 'gender': 0.65, 'time': 0.55, 'longevity': 0.80},
    'aldehydes':             {'formality': 0.70, 'season': 0.45, 'frequency': 0.55, 'gender': 0.30, 'time': 0.55, 'longevity': 0.20},
    'amberwood':             {'formality': 0.70, 'season': 0.80, 'frequency': 0.65, 'gender': 0.60, 'time': 0.80, 'longevity': 0.90},
    'orchid':                {'formality': 0.60, 'season': 0.40, 'frequency': 0.55, 'gender': 0.20, 'time': 0.55, 'longevity': 0.50},
    'bourbon vanilla':       {'formality': 0.45, 'season': 0.85, 'frequency': 0.45, 'gender': 0.30, 'time': 0.70, 'longevity': 0.90},
    'elemi resin':           {'formality': 0.65, 'season': 0.70, 'frequency': 0.65, 'gender': 0.55, 'time': 0.70, 'longevity': 0.75},
    'tobacco':               {'formality': 0.70, 'season': 0.70, 'frequency': 0.70, 'gender': 0.80, 'time': 0.80, 'longevity': 0.80},
    'tangerine':             {'formality': 0.15, 'season': 0.10, 'frequency': 0.10, 'gender': 0.45, 'time': 0.10, 'longevity': 0.10},
    'lime':                  {'formality': 0.10, 'season': 0.05, 'frequency': 0.10, 'gender': 0.50, 'time': 0.05, 'longevity': 0.10},
    'bulgarian rose':        {'formality': 0.70, 'season': 0.45, 'frequency': 0.60, 'gender': 0.10, 'time': 0.60, 'longevity': 0.55},
    'mimosa':                {'formality': 0.50, 'season': 0.30, 'frequency': 0.40, 'gender': 0.20, 'time': 0.40, 'longevity': 0.40},
    'coconut':               {'formality': 0.10, 'season': 0.10, 'frequency': 0.15, 'gender': 0.25, 'time': 0.20, 'longevity': 0.30},
    'pine':                  {'formality': 0.30, 'season': 0.70, 'frequency': 0.25, 'gender': 0.65, 'time': 0.35, 'longevity': 0.45},
    'peppermint':            {'formality': 0.10, 'season': 0.05, 'frequency': 0.10, 'gender': 0.55, 'time': 0.05, 'longevity': 0.10},
    'juniper':               {'formality': 0.40, 'season': 0.55, 'frequency': 0.35, 'gender': 0.65, 'time': 0.40, 'longevity': 0.40},
    'apricot':               {'formality': 0.20, 'season': 0.20, 'frequency': 0.20, 'gender': 0.20, 'time': 0.25, 'longevity': 0.20},
    'cypress':               {'formality': 0.50, 'season': 0.60, 'frequency': 0.45, 'gender': 0.65, 'time': 0.50, 'longevity': 0.55},
    'ambroxan':              {'formality': 0.50, 'season': 0.60, 'frequency': 0.50, 'gender': 0.55, 'time': 0.65, 'longevity': 0.90},
    'almond':                {'formality': 0.30, 'season': 0.70, 'frequency': 0.35, 'gender': 0.30, 'time': 0.55, 'longevity': 0.50},
    'eucalyptus':            {'formality': 0.15, 'season': 0.25, 'frequency': 0.15, 'gender': 0.55, 'time': 0.15, 'longevity': 0.20},
    'carnation':             {'formality': 0.60, 'season': 0.45, 'frequency': 0.55, 'gender': 0.35, 'time': 0.55, 'longevity': 0.50},
    'yuzu':                  {'formality': 0.20, 'season': 0.10, 'frequency': 0.10, 'gender': 0.45, 'time': 0.10, 'longevity': 0.10},
    'coffee':                {'formality': 0.40, 'season': 0.65, 'frequency': 0.45, 'gender': 0.60, 'time': 0.70, 'longevity': 0.60},
    'chamomile':             {'formality': 0.25, 'season': 0.35, 'frequency': 0.20, 'gender': 0.35, 'time': 0.25, 'longevity': 0.25},

    # ── Regional/varietal synonyms seen in corpus ─────────────────────────
    'italian lemon':         {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.45, 'time': 0.05, 'longevity': 0.10},
    'sicilian lemon':        {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.45, 'time': 0.05, 'longevity': 0.10},
    'calabrian bergamot':    {'formality': 0.30, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},
    'italian bergamot':      {'formality': 0.30, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},
    'haitian vetiver':       {'formality': 0.65, 'season': 0.65, 'frequency': 0.55, 'gender': 0.75, 'time': 0.65, 'longevity': 0.85},
    'australian sandalwood': {'formality': 0.60, 'season': 0.70, 'frequency': 0.50, 'gender': 0.55, 'time': 0.65, 'longevity': 0.85},
    'jasmine sambac':        {'formality': 0.60, 'season': 0.45, 'frequency': 0.55, 'gender': 0.20, 'time': 0.65, 'longevity': 0.60},
    'provençal lavender':    {'formality': 0.35, 'season': 0.40, 'frequency': 0.20, 'gender': 0.60, 'time': 0.35, 'longevity': 0.35},
    'green mandarin orange': {'formality': 0.20, 'season': 0.10, 'frequency': 0.15, 'gender': 0.45, 'time': 0.10, 'longevity': 0.15},

    # ── Broad catch-all descriptor notes seen in corpus ───────────────────
    'woody notes':           {'formality': 0.55, 'season': 0.65, 'frequency': 0.50, 'gender': 0.60, 'time': 0.60, 'longevity': 0.75},
    'green notes':           {'formality': 0.20, 'season': 0.25, 'frequency': 0.15, 'gender': 0.45, 'time': 0.20, 'longevity': 0.20},
    'floral notes':          {'formality': 0.45, 'season': 0.35, 'frequency': 0.35, 'gender': 0.15, 'time': 0.40, 'longevity': 0.40},
    'citrus notes':          {'formality': 0.15, 'season': 0.05, 'frequency': 0.10, 'gender': 0.45, 'time': 0.05, 'longevity': 0.10},
    'spicy notes':           {'formality': 0.60, 'season': 0.65, 'frequency': 0.55, 'gender': 0.55, 'time': 0.65, 'longevity': 0.50},
    'fruity notes':          {'formality': 0.15, 'season': 0.20, 'frequency': 0.20, 'gender': 0.20, 'time': 0.20, 'longevity': 0.20},
    'spices':                {'formality': 0.60, 'season': 0.70, 'frequency': 0.55, 'gender': 0.55, 'time': 0.65, 'longevity': 0.45},
    'woods':                 {'formality': 0.55, 'season': 0.65, 'frequency': 0.50, 'gender': 0.60, 'time': 0.60, 'longevity': 0.80},
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