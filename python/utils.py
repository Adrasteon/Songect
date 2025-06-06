import torch.nn.functional as F
from common import CHORD_LABELS, SEGMENT_LABELS
import whisper
import torch
import numpy as np
import librosa # Added for resampling
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC # Ensure this is at the top with other imports


def build_masked_stft(masks, stft_feature, n_fft=4096):
    out = []
    for i in range(len(masks)):
        mask = masks[i, :, :, :]
        pad_num = n_fft // 2 + 1 - mask.size(-1)
        mask = F.pad(mask, (0, pad_num, 0, 0, 0, 0))
        inst_stft = mask.type(stft_feature.dtype) * stft_feature
        out.append(inst_stft)
    return out


def get_chord_name(chord_idx_list):
    chords = [CHORD_LABELS[idx] for idx in chord_idx_list]
    return chords


def get_segment_name(segments):
    segments = [SEGMENT_LABELS[idx] for idx in segments]
    return segments


# Helper functions for Wav2Vec2 alignment (inspired by PyTorch Forced Alignment Tutorial)
# IMPORTANT: These helper functions are simplified placeholders and need careful implementation
# by adapting the PyTorch tutorial or other robust alignment logic.

def get_trellis(emission, tokens, blank_id=0):
    # emission: (num_frames, num_classes) - Log probabilities from Wav2Vec2 model
    # tokens: List of original token IDs for the transcript (e.g., [t1, t2, t3])
    # blank_id: ID of the blank token in the model's vocabulary
    # This function computes and returns the CTC trellis (log probabilities).

    num_frames = emission.size(0)
    device = emission.device

    # 1. Construct ctc_token_ids: [blank, t1, blank, t2, blank, ..., tn, blank]
    ctc_token_ids = [blank_id]
    for token_id in tokens:
        ctc_token_ids.append(token_id)
        ctc_token_ids.append(blank_id)
    num_ctc_tokens = len(ctc_token_ids)

    # 2. Initialize trellis
    # trellis[t, j] = log probability of being in ctc_token_ids[j] at time t
    trellis = torch.full((num_frames, num_ctc_tokens), -float('inf'), device=device)

    # 3. Set initial probabilities at t=0
    # trellis[0, 0] = p(blank | frame 0)
    trellis[0, 0] = emission[0, blank_id]
    if num_ctc_tokens > 1:
        # trellis[0, 1] = p(tokens[0] | frame 0)
        trellis[0, 1] = emission[0, ctc_token_ids[1]] # ctc_token_ids[1] is tokens[0]

    # 4. Iterate through time and ctc_tokens to fill the trellis
    for t in range(1, num_frames):
        for j in range(num_ctc_tokens):
            current_ctc_token_vocab_id = ctc_token_ids[j]
            log_p_emit = emission[t, current_ctc_token_vocab_id]

            # Probability of coming from the same ctc_token_ids[j] at t-1
            p_stay = trellis[t-1, j]
            
            # Probability of coming from ctc_token_ids[j-1] at t-1
            p_prev = trellis[t-1, j-1] if j > 0 else -float('inf')

            # CTC recurrence relations
            if current_ctc_token_vocab_id == blank_id or (j > 1 and ctc_token_ids[j] == ctc_token_ids[j-2]):
                # If current token is blank, or it's a repeat of the token before the previous blank
                # (e.g., path B -> T1 -> B -> T1, current is the second T1)
                # Allowed transitions: from same state (j) or previous state (j-1)
                log_probs_sum = torch.logsumexp(torch.tensor([p_stay, p_prev], device=device), dim=0)
            else:
                # If current token is a non-blank and different from token before previous blank
                # (e.g., path B -> T1 -> B -> T2, current is T2)
                # Allowed transitions: from same state (j), previous state (j-1), or state before previous blank (j-2)
                p_prev_prev_token = trellis[t-1, j-2] if j > 1 else -float('inf')
                log_probs_sum = torch.logsumexp(torch.tensor([p_stay, p_prev, p_prev_prev_token], device=device), dim=0)
            
            trellis[t, j] = log_probs_sum + log_p_emit
            
    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    """
    Backtracks through the trellis to find the most probable path of token emissions.

    Args:
        trellis (torch.Tensor): The CTC trellis of shape (num_frames, num_ctc_tokens).
        emission (torch.Tensor): Log probabilities from the Wav2Vec2 model (num_frames, num_classes).
                                 Not strictly used in this backtracking version if trellis incorporates emissions,
                                 but kept for signature consistency with some tutorials.
        tokens (list): List of original token IDs for the transcript (e.g., [t1, t2, t3]).
        blank_id (int): ID of the blank token in the model's vocabulary.

    Returns:
        list: A list of actual vocabulary token IDs representing the most probable path.
    """
    num_frames = trellis.size(0)
    device = trellis.device

    # 1. Construct ctc_token_ids (same as in get_trellis)
    ctc_token_ids = [blank_id]
    for token_id in tokens:
        ctc_token_ids.append(token_id)
        ctc_token_ids.append(blank_id)
    num_ctc_tokens = len(ctc_token_ids)

    # Initialize best_path with indices into ctc_token_ids
    best_path_indices = torch.zeros(num_frames, dtype=torch.long, device=device)

    # 2. Determine the end of the path at t = num_frames - 1
    # Path must end in the last ctc_token (blank) or the one before it (last actual token)
    if trellis[num_frames - 1, num_ctc_tokens - 1] > trellis[num_frames - 1, num_ctc_tokens - 2]:
        best_path_indices[num_frames - 1] = num_ctc_tokens - 1
    else:
        best_path_indices[num_frames - 1] = num_ctc_tokens - 2

    # 3. Iterate backwards from t = num_frames - 2 down to 0
    for t in range(num_frames - 2, -1, -1):
        # j_next is the index in ctc_token_ids chosen for best_path_indices[t+1]
        # Explicitly cast .item() to int for clarity for the linter.
        j_next = int(best_path_indices[t + 1].item())

        # Candidate probabilities at time t for states that could lead to j_next
        val_at_j_next = trellis[t, j_next]
        val_at_j_next_minus_1 = trellis[t, j_next-1] if j_next > 0 else -float('inf')

        # j_next is now explicitly an int.
        is_blank_or_repeat_at_j_next = (ctc_token_ids[j_next] == blank_id) or \
                                        (j_next >= 2 and ctc_token_ids[j_next] == ctc_token_ids[j_next-2])

        if is_blank_or_repeat_at_j_next:
            # If j_next is blank or a repeat, it could only have been reached from j_next or j_next-1 at time t.
            if val_at_j_next >= val_at_j_next_minus_1:
                best_path_indices[t] = j_next
            else:
                best_path_indices[t] = j_next-1
        else:
            # If j_next is a non-blank and non-repeat, it could have been reached from j_next, j_next-1, or j_next-2 at time t.
            val_at_j_next_minus_2 = trellis[t, j_next-2] if j_next > 1 else -float('inf')
            
            if val_at_j_next >= val_at_j_next_minus_1 and val_at_j_next >= val_at_j_next_minus_2:
                best_path_indices[t] = j_next
            elif val_at_j_next_minus_1 >= val_at_j_next and val_at_j_next_minus_1 >= val_at_j_next_minus_2:
                best_path_indices[t] = j_next-1
            else:
                best_path_indices[t] = j_next-2
    
    # 4. Convert path of ctc_token_indices to actual vocabulary token IDs
    # Explicitly cast idx.item() to int for clarity for the linter.
    actual_path_vocab_ids = [ctc_token_ids[int(idx.item())] for idx in best_path_indices]
    return actual_path_vocab_ids

def merge_repeats_and_blanks(path, ratio, tokenizer, blank_id=0):
    """
    Merges consecutive repeated non-blank tokens and removes blank tokens from the path.
    Calculates start and end times for each resulting token segment.

    Args:
        path (list): List of token IDs from backtrack (one token ID per frame).
        ratio (float): Time duration of a single frame.
        tokenizer: Wav2Vec2 tokenizer (currently not used in this function but kept for potential future use
                   or consistency with other parts of alignment pipelines).
        blank_id (int): ID of the blank token.

    Returns:
        list: A list of dictionaries, where each dictionary represents a token segment
              and has keys: 'token_id', 'start_time', 'end_time'.
    """
    segments = []
    i = 0
    while i < len(path):
        token_id = path[i]

        # Skip blank tokens
        if token_id == blank_id:
            i += 1
            continue

        # Current token is non-blank, find its extent
        start_frame = i
        while i + 1 < len(path) and path[i+1] == token_id:
            i += 1
        end_frame = i # end_frame is the index of the last frame containing this token_id

        # Ensure token_id is a Python int if it's a tensor element
        processed_token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id

        segments.append({
            "token_id": processed_token_id,
            "start_time": round(start_frame * ratio, 3),
            "end_time": round((end_frame + 1) * ratio, 3) # End time is exclusive of the segment
        })
        
        i += 1
        
    # print(f"[WIP] merge_repeats_and_blanks: Produced {len(segments)} token segments.") # Removed WIP print
    return segments

def _normalize_text_for_alignment(text: str) -> str:
    """Helper to normalize text for alignment comparison."""
    if not isinstance(text, str):
        return ""
    # Lowercase, strip whitespace, and remove some common punctuation/characters for robust matching.
    # This normalization should be tailored to the specific tokenizer and expected transcript variations.
    text = text.lower().strip()
    text = text.replace("'", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace(" ", "") # Also remove internal spaces for very robust matching against concatenated subwords
    return text

def merge_tokens_to_words_from_alignment(token_segments, original_transcript_words, tokenizer):
    """
    Merges timed token segments into words, aligning them with the original transcript words.

    Args:
        token_segments (list): List of {'token_id': id, 'start_time': s, 'end_time': e}.
        original_transcript_words (list): List of words (strings) from the ASR transcript.
        tokenizer: The Wav2Vec2 tokenizer instance.

    Returns:
        list: A list of dictionaries, {'word': str, 'start_time': float, 'end_time': float}.
    """
    if not token_segments or not original_transcript_words:
        print("merge_tokens_to_words_from_alignment: Empty token_segments or original_transcript_words.")
        return []

    word_segments = []
    token_idx = 0  # Main cursor for token_segments

    for word_idx_orig, target_word_orig in enumerate(original_transcript_words):
        if token_idx >= len(token_segments):
            # All tokens have been consumed, but there are still words in the transcript.
            # print(f"Ran out of tokens while trying to align word: '{target_word_orig}' (index {word_idx_orig})")
            break

        current_word_start_time = token_segments[token_idx]['start_time']
        accumulated_token_ids_for_current_word = []
        found_match_for_target_word = False

        # Try to match target_word_orig using tokens starting from token_idx
        for search_token_idx in range(token_idx, len(token_segments)):
            current_token_id = token_segments[search_token_idx]['token_id']
            accumulated_token_ids_for_current_word.append(current_token_id)

            reconstructed_word_str = tokenizer.decode(
                accumulated_token_ids_for_current_word,
                skip_special_tokens=True
            )

            normalized_reconstructed = _normalize_text_for_alignment(reconstructed_word_str)
            normalized_target = _normalize_text_for_alignment(target_word_orig)

            if not normalized_target: # Should not happen if original_transcript_words is clean
                # print(f"Skipping empty target word at index {word_idx_orig}.")
                found_match_for_target_word = True # Effectively skip it
                break

            if normalized_reconstructed == normalized_target:
                word_segments.append({
                    "word": target_word_orig, # Use original casing from transcript
                    "start_time": current_word_start_time,
                    "end_time": token_segments[search_token_idx]['end_time']
                })
                token_idx = search_token_idx + 1  # Consume these tokens
                found_match_for_target_word = True
                break  # Move to the next target_word_orig
            elif len(normalized_reconstructed) > 0 and not normalized_target.startswith(normalized_reconstructed):
                # The reconstructed string has diverged from the target word.
                # Stop accumulating tokens for *this* target_word_orig with *this* starting token sequence.
                break
            # Else (target starts with reconstructed, or reconstructed is empty), continue accumulating (next search_token_idx)
            # or if reconstructed is empty and target is not, it will eventually fail the startswith or match.

        if not found_match_for_target_word:
            # print(f"Warning: Could not align word '{target_word_orig}' (index {word_idx_orig}) with tokens starting at token_segments index {token_idx}.")
            pass # Current token_idx remains, try matching next original word with current tokens.

    # print(f"merge_tokens_to_words_from_alignment: Produced {len(word_segments)} word segments.")
    return word_segments


def get_lyrics(waveform, sr, cfg):
    """
    Transcribes lyrics and performs word-level alignment.

    Args:
        waveform (torch.Tensor): The audio waveform.
        sr (int): The sample rate of the waveform.
        cfg (dict): Configuration dictionary, potentially containing:
                    'whisper_model': Name of the Whisper model.
                    'wav2vec2_alignment_model': Name of the Wav2Vec2 model for alignment.
                    'alignment_device': Device for alignment model ('cpu', 'cuda').

    Returns:
        dict: A dictionary containing:
              'text' (str): The full transcribed text.
              'detailed_lyrics' (list): A list of dictionaries, where each dict has
                                        'word', 'start_time', 'end_time'.
              Returns None or a dict with empty fields if errors occur.
    """
    transcribed_text = ""
    whisper_word_segments = None
    detailed_lyrics_data = [] # Initialize here
    waveform_np_float32 = None # Initialize to ensure it's always bound

    # --- Part 1: ASR with Whisper ---
    try:
        model_name_whisper = cfg.get('whisper_model', 'base')
        print(f"Loading Whisper model: {model_name_whisper}...")
        model_whisper = whisper.load_model(model_name_whisper)
        print("Whisper model loaded.")

        if waveform.ndim > 1:
            if waveform.shape[0] == 2: # Stereo
                waveform = torch.mean(waveform, dim=0)
            else: # More than 2 channels
                waveform = waveform[0, :]
        
        waveform_1d = waveform.squeeze()
        if waveform_1d.ndim == 0:
            waveform_1d = waveform_1d.unsqueeze(0)
        waveform_np_float32 = waveform_1d.cpu().numpy().astype(np.float32)

        print("Starting transcription with Whisper...")
        whisper_result = model_whisper.transcribe(waveform_np_float32,
                                                  fp16=torch.cuda.is_available(),
                                                  word_timestamps=True) # Get Whisper's word timestamps
        print("Whisper transcription complete.")
        transcribed_text = whisper_result["text"]
        # Linter fix: Ensure transcribed_text is a string
        if not isinstance(transcribed_text, str):
            if isinstance(transcribed_text, list):
                transcribed_text = " ".join(map(str, transcribed_text))
            else:
                transcribed_text = str(transcribed_text) # Fallback

        whisper_word_segments = whisper_result.get("words") # Use .get() for safety

    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return {"text": f"Whisper ASR Error: {e}", "detailed_lyrics": []}

    # Check if Whisper produced any output before proceeding
    if not transcribed_text.strip() and not whisper_word_segments:
        print("Whisper produced no text or word segments.")
        return {"text": "", "detailed_lyrics": []}

    # --- Part 2: Process Whisper's word segments (or prepare for Wav2Vec2 alignment) ---
    try:
        # Initialize with Whisper's word timestamps
        if whisper_word_segments:
            print("Processing Whisper word segments for initial detailed_lyrics.")
            for segment in whisper_word_segments:
                # Linter fix: Ensure segment is a dict
                if not isinstance(segment, dict):
                    print(f"Warning: Expected segment to be a dict, but got {type(segment)}. Value: {segment}. Skipping.")
                    continue

                word_text = segment.get('word', '').strip()
                start_time = segment.get('start')
                end_time = segment.get('end')

                if word_text and start_time is not None and end_time is not None:
                    # Only add if word, start, and end times are present
                    detailed_lyrics_data.append({
                        "word": word_text, 
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3)
                    })
                elif word_text:
                    # Optional: Log a warning if word is present but timings are missing
                    print(f"Warning: Segment for word '{word_text}' is missing start and/or end time. Segment data: {segment}")
        elif transcribed_text.strip():
            print("Whisper did not provide word timestamps, but text is available. Creating basic segments.")
            words = transcribed_text.split()
            current_time = 0.0
            for word in words:
                duration = 0.5 # Arbitrary duration
                detailed_lyrics_data.append({
                    "word": word,
                    "start_time": round(current_time, 2),
                    "end_time": round(current_time + duration, 2)
                })
                current_time += duration + 0.05 # Arbitrary gap
        else:
            # This case should ideally be caught by the check after Whisper transcription
            print("No text from Whisper to process for detailed lyrics.")
            # detailed_lyrics_data remains empty, which is fine

    except Exception as e:
        print(f"Error during word segmentation/alignment stage: {e}")
        # Fallback: if alignment fails, try to construct from Whisper's text if not already done
        # and if detailed_lyrics_data is still empty
        if not detailed_lyrics_data and transcribed_text:
            print("Error in alignment, falling back to basic text split for detailed_lyrics.")
            words = transcribed_text.split()
            current_time = 0.0
            for word in words:
                duration = 0.5
                gap = 0.05
                detailed_lyrics_data.append({"word": word, "start_time": round(current_time,2), "end_time": round(current_time+duration,2)})
                current_time += duration + gap
        # else, detailed_lyrics_data might be from Whisper's attempt or empty

    # --- Part 2: Wav2Vec2 Alignment Refinement ---
    
    use_wav2vec2_alignment = cfg.get('use_wav2vec2_alignment', False) 

    if use_wav2vec2_alignment and transcribed_text.strip():
        print("Attempting Wav2Vec2 alignment refinement...")
        try:
            # Ensure waveform_np_float32 is available from Whisper processing
            if waveform_np_float32 is None:
                raise ValueError("Audio data (waveform_np_float32) is missing for Wav2Vec2 processing.")

            alignment_model_name = cfg.get('wav2vec2_alignment_model', 'facebook/wav2vec2-base-960h')
            alignment_device = cfg.get('alignment_device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"Loading Wav2Vec2 Processor: {alignment_model_name}...")
            processor_val = Wav2Vec2Processor.from_pretrained(alignment_model_name)
            # Linter fix: Handle if from_pretrained returns a tuple or unexpected type
            if isinstance(processor_val, tuple):
                if processor_val and len(processor_val) > 0 and isinstance(processor_val[0], Wav2Vec2Processor):
                    processor = processor_val[0]
                else:
                    raise TypeError(f"Wav2Vec2Processor.from_pretrained returned an unexpected tuple: {processor_val}")
            elif isinstance(processor_val, Wav2Vec2Processor):
                processor = processor_val
            else:
                raise TypeError(f"Wav2Vec2Processor.from_pretrained returned an unexpected type: {type(processor_val)}")

            print(f"Loading Wav2Vec2 Model: {alignment_model_name} for CTC...")
            model = Wav2Vec2ForCTC.from_pretrained(alignment_model_name).to(alignment_device)

            # Access 'feature_extractor' using getattr to potentially resolve linter issues
            # with the type definition of Wav2Vec2Processor.
            # The linter previously reported: Cannot access attribute "feature_extractor" for class "Wav2Vec2Processor"
            _feature_extractor_object = getattr(processor, "feature_extractor")
            target_sr_wav2vec2 = _feature_extractor_object.sampling_rate

            # waveform_np_float32 is guaranteed not None here due to the check above
            waveform_np_resampled = waveform_np_float32 
            if sr != target_sr_wav2vec2:
                print(f"Resampling audio from {sr}Hz to {target_sr_wav2vec2}Hz for Wav2Vec2...")
                waveform_np_resampled = librosa.resample(waveform_np_float32, orig_sr=sr, target_sr=target_sr_wav2vec2)
            
            print("Processing audio for Wav2Vec2...")
            # waveform_np_resampled is now guaranteed to be a numpy array if no exception was raised.
            input_values = processor(waveform_np_resampled, sampling_rate=target_sr_wav2vec2, return_tensors="pt", padding=True).input_values.to(alignment_device)
            
            print("Getting logits from Wav2Vec2...")
            with torch.no_grad():
                logits = model(input_values).logits.cpu()[0] # Logits for the first batch item

            # The PyTorch tutorial's full alignment logic is complex.
            # The following is a placeholder for that logic.
            # For a real implementation, adapt the tutorial's `get_trellis`, `backtrack`,
            # `merge_repeats`, and `merge_words` (our `merge_tokens_to_words_from_alignment`).
            
            # --- Placeholder for actual alignment logic from PyTorch tutorial ---
            # 1. Prepare transcript tokens for CTC alignment
            # Ensure transcribed_text is not empty before tokenizing
            if not transcribed_text.strip():
                print("Skipping Wav2Vec2 alignment as transcribed_text is empty.")
            else:
                # Access tokenizer using getattr for robustness with linters
                tokenizer_object = getattr(processor, "tokenizer")
                if tokenizer_object is None:
                    raise AttributeError("Wav2Vec2Processor does not have a 'tokenizer' attribute or it is None.")

                # Tokenize the full transcript for CTC alignment target.
                tokens_for_alignment = tokenizer_object(transcribed_text, add_special_tokens=False).input_ids
                
                # Blank ID for CTC is typically the pad_token_id for Wav2Vec2
                blank_id = tokenizer_object.pad_token_id
                if blank_id is None:
                    print("Warning: Wav2Vec2 tokenizer.pad_token_id is None. Attempting to use 0 as blank_id.")
                    blank_id = 0 

                # 2. Compute Trellis & Backtrack (using the robust helper functions)
                print("Computing CTC trellis...")
                trellis = get_trellis(logits, tokens_for_alignment, blank_id)
                print("Backtracking through trellis...")
                path = backtrack(trellis, logits, tokens_for_alignment, blank_id) # list of vocab token_ids

                # 3. Calculate time ratio for frames
                # Total duration of the audio processed by Wav2Vec2 (which might be resampled)
                audio_duration_sec = waveform_np_resampled.shape[0] / target_sr_wav2vec2
                num_logit_frames = logits.size(0)
                if num_logit_frames == 0:
                    raise ValueError("Logits have zero frames, cannot calculate time_ratio.")
                time_ratio = audio_duration_sec / num_logit_frames # Time per logit frame

                # 4. Merge repeated tokens/blanks from path to get token-level segments
                print("Merging repeats and blanks from path...")
                token_segments = merge_repeats_and_blanks(path, time_ratio, tokenizer_object, blank_id)
                
                # 5. Merge token segments into word segments
                original_words_for_merging = transcribed_text.split()
                if not original_words_for_merging:
                    print("Warning: Transcribed text resulted in no words for merging.")
                    refined_word_segments = []
                else:
                    print("Merging token segments into words...")
                    refined_word_segments = merge_tokens_to_words_from_alignment(
                        token_segments,
                        original_words_for_merging, 
                        tokenizer_object
                    )

                if refined_word_segments:
                    detailed_lyrics_data = refined_word_segments # Replace Whisper's data
                    print("Wav2Vec2 alignment applied successfully.")
                else:
                    print("Wav2Vec2 alignment did not produce refined word segments. Using Whisper's timestamps.")
            # --- End Actual Alignment Logic ---

        except Exception as e_wav2vec:
            print(f"Error during Wav2Vec2 alignment execution: {e_wav2vec}. Using Whisper's timestamps.")
            # In case of error, we continue with the data from Whisper (already in detailed_lyrics_data)

    return {
        "text": transcribed_text if transcribed_text else "",
        "detailed_lyrics": detailed_lyrics_data
    }
