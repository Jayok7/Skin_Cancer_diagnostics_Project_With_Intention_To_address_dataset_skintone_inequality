"""
quality_control.py
==================
Five-stage quality gate for teacher-generated clinical reasoning.

Stages:
  1. Keyword constraint checking - min 2 clinical keyword matches
  2. Indeterminate filter          - reject hedged non-answers
  3. Spatial alignment (Grad-CAM)  - spatial references must match heatmap
  4. Length check                   - 60-200 words
  5. Safety language               - must hedge, must not be definitive

Expected discard rate: 20-30%  →  from 2500 raw → target ≥1750 clean
Minimum viable dataset: 1500 clean examples

Usage:
    python quality_control.py \
        --input  training_data/teacher_outputs_free.json \
        --output training_data/clean_dataset.json \
        --report training_data/qc_report.json
"""

import json
import argparse
from collections import Counter


# ──────────────────────────────────────────────
# Stage 1: Keyword Constraint Checking
# ──────────────────────────────────────────────
def check_keywords(record, min_keyword_matches=2):
    """
    Verify that the teacher output contains at least `min_keyword_matches`
    keywords from the expected keyword list for the predicted class.

    Why: catches gross factual errors - e.g., if the CNN predicts melanoma
    but the teacher describes "cherry angioma," something is wrong.

    Returns: (pass: bool, matched_keywords: list)
    """
    reasoning = record['teacher_reasoning'].lower()

    # Use explicit expected_keywords if available, otherwise derive from class
    if 'expected_keywords' in record:
        expected = record['expected_keywords']
    else:
        # Fallback: derive keywords from predicted class name
        expected = _get_class_keywords(record.get('predicted_label', record.get('predicted_name', '')))

    matched = [kw for kw in expected if kw.lower() in reasoning]

    return len(matched) >= min_keyword_matches, matched


# Class-specific keyword fallback map
_CLASS_KEYWORD_MAP = {
    'mel': ['melanoma', 'pigment', 'asymmetry', 'border', 'atypical', 'nevus', 'dermoscopic'],
    'melanoma': ['melanoma', 'pigment', 'asymmetry', 'border', 'atypical', 'nevus', 'dermoscopic'],
    'nv': ['nevus', 'nevi', 'melanocytic', 'pigment', 'benign', 'regular', 'symmetric'],
    'melanocytic nevus': ['nevus', 'nevi', 'melanocytic', 'pigment', 'benign', 'regular'],
    'bcc': ['basal cell', 'carcinoma', 'pearlescent', 'telangiectasia', 'nodular', 'ulcer'],
    'basal cell carcinoma': ['basal cell', 'carcinoma', 'pearlescent', 'telangiectasia'],
    'akiec': ['actinic', 'keratosis', 'bowen', 'squamous', 'scaly', 'hyperkeratotic'],
    'bkl': ['keratosis', 'seborrheic', 'benign', 'pigmented', 'dermatosis'],
    'df': ['dermatofibroma', 'fibrous', 'dimple', 'central', 'papule'],
    'vasc': ['vascular', 'angioma', 'cherry', 'hemangioma', 'blood', 'vessel'],
    'scc': ['squamous', 'cell', 'carcinoma', 'keratinizing', 'ulcerated'],
    'ak': ['actinic', 'keratosis', 'scaly', 'rough', 'solar'],
}

def _get_class_keywords(class_name):
    """Get fallback keywords for a class name."""
    key = class_name.lower().strip()
    return _CLASS_KEYWORD_MAP.get(key, ['lesion', 'skin', 'dermoscopic'])


# ──────────────────────────────────────────────
# Stage 2: Indeterminate Handling
# ──────────────────────────────────────────────
INDETERMINATE_PHRASES = [
    'indeterminate',
    'cannot be determined',
    'unable to identify',
    'not visible',
    'insufficient resolution',
    'too low to assess',
    'cannot clearly see',
    'features are unclear'
]

def check_indeterminate(record):
    """
    If the teacher states features are indeterminate, discard the example.

    Why: prevents the student model from learning to produce hedged
    non-answers. We want every training example to contain substantive
    reasoning. The student will learn hedging behavior via DPO instead.

    Returns: (pass: bool, matched_phrase: str or None)
    """
    reasoning = record['teacher_reasoning'].lower()

    for phrase in INDETERMINATE_PHRASES:
        if phrase in reasoning:
            return False, phrase

    return True, None


# ──────────────────────────────────────────────
# Stage 3: Spatial Alignment with Grad-CAM
# ──────────────────────────────────────────────
SPATIAL_TERMS = {
    'upper':  ['upper', 'top', 'superior'],
    'lower':  ['lower', 'bottom', 'inferior'],
    'left':   ['left', 'lateral left'],
    'right':  ['right', 'lateral right'],
    'center': ['center', 'central', 'middle', 'core'],
    'diffuse':['diffuse', 'widespread', 'throughout', 'entire', 'broadly']
}

def extract_spatial_references(text):
    """Extract spatial terms mentioned in the reasoning text"""
    text_lower = text.lower()
    found_regions = set()

    for region, terms in SPATIAL_TERMS.items():
        for term in terms:
            if term in text_lower:
                found_regions.add(region)

    return found_regions


def check_spatial_alignment(record, tolerance=True):
    """
    Verify that spatial references in the teacher's text align with
    where the Grad-CAM heatmap actually activates.

    Why: ensures the reasoning is spatially faithful to the CNN's actual
    attention pattern, not fabricated. If the teacher says "the upper-left
    region shows irregular pigmentation" but Grad-CAM highlights the center,
    the example is discarded.

    tolerance: if True, allows 'center' to be compatible with adjacent regions

    Returns: (pass: bool, reason: str)
    """
    reasoning = record['teacher_reasoning']

    if 'gradcam_spatial_region' not in record:
        return True, "No spatial region in metadata, assuming pass"

    gradcam_region = record['gradcam_spatial_region']  # e.g., 'center', 'upper-left'

    # Parse Grad-CAM region into component parts
    gradcam_parts = set(gradcam_region.split('-'))
    if gradcam_region == 'diffuse':
        # Diffuse activation is compatible with any spatial reference
        return True, 'diffuse activation: any spatial reference acceptable'

    # Extract spatial references from text
    text_regions = extract_spatial_references(reasoning)

    if len(text_regions) == 0:
        # No spatial claims made, passes by default
        return True, 'no spatial claims in text'

    # Check for contradictions
    contradictions = []

    # Define opposing regions
    opposites = {
        'upper': 'lower', 'lower': 'upper',
        'left': 'right', 'right': 'left'
    }

    for text_region in text_regions:
        if text_region == 'diffuse' or text_region == 'center':
            continue  # These are generally compatible

        # Check if text claims a region opposite to Grad-CAM
        if text_region in opposites:
            opposite = opposites[text_region]
            if opposite in gradcam_parts and text_region not in gradcam_parts:
                contradictions.append(
                    f'Text says "{text_region}" but Grad-CAM shows "{gradcam_region}"'
                )

    if contradictions:
        return False, '; '.join(contradictions)

    return True, 'spatial references aligned'


# ──────────────────────────────────────────────
# Stage 4: Length Check
# ──────────────────────────────────────────────
def check_length(record, min_words=60, max_words=200):
    """Ensure response is within acceptable length range"""
    word_count = len(record['teacher_reasoning'].split())
    return min_words <= word_count <= max_words, word_count


# ──────────────────────────────────────────────
# Stage 5: Safety Language
# ──────────────────────────────────────────────
def check_safety_language(record):
    """
    Ensure the response contains hedging language and doesn't make
    definitive diagnostic claims.
    """
    reasoning = record['teacher_reasoning'].lower()

    # Must contain at least one hedging phrase
    hedging_phrases = [
        'consistent with', 'suggestive of', 'characteristic of',
        'may represent', 'features of', 'indicative of',
        'correlate', 'biopsy', 'clinical', 'recommend',
        'should be considered', 'differential'
    ]

    has_hedging = any(phrase in reasoning for phrase in hedging_phrases)

    # Must NOT contain definitive language
    definitive_phrases = [
        'this is definitely', 'this is certainly',
        'i diagnose', 'the diagnosis is confirmed',
        'no doubt', 'without question'
    ]

    has_definitive = any(phrase in reasoning for phrase in definitive_phrases)

    if has_definitive:
        return False, 'contains definitive diagnostic language'
    if not has_hedging:
        return False, 'missing hedging/safety language'

    return True, 'appropriate clinical language'


# ──────────────────────────────────────────────
# Main QC Pipeline
# ──────────────────────────────────────────────
def run_quality_control(input_path, output_path, report_path):
    """
    Run the full quality control pipeline.

    Expected discard rate: 20-30%
    Minimum target: 1500 clean training examples
    Ideal target: 1750-2000 clean training examples
    """

    with open(input_path, 'r') as f:
        records = json.load(f)

    total = len(records)
    clean = []
    discarded = {
        'keyword_fail': [],
        'indeterminate': [],
        'spatial_misalign': [],
        'length_fail': [],
        'safety_fail': []
    }

    for record in records:
        # Stage 1: Keyword check
        kw_pass, kw_matched = check_keywords(record)
        if not kw_pass:
            discarded['keyword_fail'].append({
                'image_id': record['image_id'],
                'predicted': record['predicted_class'],
                'matched': kw_matched,
                'expected': record['expected_keywords']
            })
            continue

        # Stage 2: Indeterminate check
        indet_pass, indet_phrase = check_indeterminate(record)
        if not indet_pass:
            discarded['indeterminate'].append({
                'image_id': record['image_id'],
                'phrase': indet_phrase
            })
            continue

        # Stage 3: Spatial alignment
        spatial_pass, spatial_reason = check_spatial_alignment(record)
        if not spatial_pass:
            discarded['spatial_misalign'].append({
                'image_id': record['image_id'],
                'reason': spatial_reason,
                'gradcam_region': record['gradcam_spatial_region']
            })
            continue

        # Stage 4: Length check
        len_pass, word_count = check_length(record)
        if not len_pass:
            discarded['length_fail'].append({
                'image_id': record['image_id'],
                'word_count': word_count
            })
            continue

        # Stage 5: Safety language
        safety_pass, safety_reason = check_safety_language(record)
        if not safety_pass:
            discarded['safety_fail'].append({
                'image_id': record['image_id'],
                'reason': safety_reason
            })
            continue

        # Passed all checks, enrich with QC metadata
        record['qc_metadata'] = {
            'keywords_matched': kw_matched,
            'spatial_status': spatial_reason,
            'word_count': word_count
        }
        clean.append(record)

    # ── Report ──
    report = {
        'total_input': total,
        'total_clean': len(clean),
        'total_discarded': total - len(clean),
        'discard_rate': f'{((total - len(clean)) / max(total, 1)) * 100:.1f}%',
        'discard_breakdown': {
            k: len(v) for k, v in discarded.items()
        },
        'class_distribution': dict(Counter(r['predicted_class'] for r in clean)),
        'meets_minimum_threshold': len(clean) >= 1500,
        'target_thresholds': {
            'minimum_viable': 1500,
            'ideal_target': 1750,
            'stretch_goal': 2000
        }
    }

    # Save outputs
    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=2)

    with open(report_path, 'w') as f:
        json.dump({
            'summary': report,
            'discarded_details': discarded
        }, f, indent=2)

    # Print summary
    print('\n' + '=' * 60)
    print('QUALITY CONTROL REPORT')
    print('=' * 60)
    print(f'Total input:        {total}')
    print(f'Total clean:        {len(clean)}')
    print(f'Total discarded:    {total - len(clean)}')
    print(f'Discard rate:       {report["discard_rate"]}')
    print(f'\nBreakdown:')
    for k, v in report['discard_breakdown'].items():
        print(f'  {k:20s}: {v}')
    print(f'\nClass distribution in clean set:')
    for cls, count in sorted(report['class_distribution'].items()):
        print(f'  {cls:8s}: {count}')
    print(f'\nMeets minimum (1500): '
          f'{"YES ✓" if report["meets_minimum_threshold"] else "NO, NEEDS MORE DATA ✗"}')
    print('=' * 60)

    return clean, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quality control pipeline for teacher-generated reasoning'
    )
    parser.add_argument('--input',
                        default='training_data/teacher_outputs_free.json',
                        help='Path to teacher outputs JSON')
    parser.add_argument('--output',
                        default='training_data/clean_dataset.json',
                        help='Path for clean output dataset')
    parser.add_argument('--report',
                        default='training_data/qc_report.json',
                        help='Path for QC report')
    args = parser.parse_args()

    run_quality_control(args.input, args.output, args.report)
