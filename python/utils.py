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
    # tokens: List of token IDs for the transcript
    # blank_id: ID of the blank token in the model's vocabulary
    # This function should compute and return the CTC trellis (log probabilities)
    # Refer to PyTorch tutorial for a correct implementation.
    num_frame = emission.size(0)
    num_tokens = len(tokens) # This should be 2*len(original_tokens)+1 if using tutorial's convention
    # For now, returning a dummy trellis
    print("[WIP] get_trellis: Dummy implementation.")
    return torch.full((num_frame, num_tokens), -float('inf')).to(emission.device) # Dummy log_prob trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    # This function backtracks through the trellis to find the most probable path.
    # Refer to PyTorch tutorial.
    # Returning a dummy path (list of token_ids)
    print("[WIP] backtrack: Dummy implementation.")
    return [blank_id] * emission.size(0) # Dummy path of all blanks

def merge_repeats_and_blanks(path, ratio, tokenizer, blank_id=0):
    # path: list of token IDs from backtrack
    # ratio: time per frame
    # tokenizer: Wav2Vec2 tokenizer to decode token IDs
    # blank_id: ID of the blank token
    # This function merges consecutive repeated tokens and removes blanks.
    # It should return segments like [{'token_id': id, 'token': str, 'start': s, 'end': e}, ...]
    # Refer to PyTorch tutorial's "merge_repeats" logic.
    segments = []
    i = 0
    while i < len(path):
        token_id = path[i]
        # Decode token for debugging or if needed, though tutorial primarily works with IDs until word merging
        # token_str = tokenizer.decode([token_id], skip_special_tokens=False).strip() # Be careful with skip_special_tokens

        if token_id != blank_id:
            start_frame = i
            while i + 1 < len(path) and path[i+1] == token_id: # Merge repeats of the same token
                i += 1
            end_frame = i
            segments.append({
                "token_id": token_id.item() if isinstance(token_id, torch.Tensor) else token_id,
                "start_time": round(start_frame * ratio, 3),
                "end_time": round((end_frame + 1) * ratio, 3) # +1 because frame index is start
            })
        i += 1
    print(f"[WIP] merge_repeats_and_blanks: Produced {len(segments)} token segments.")
    return segments

def merge_tokens_to_words_from_alignment(token_segments, original_transcript_words, tokenizer):
    # token_segments: list of {'token_id': id, 'start_time': s, 'end_time': e}
    # original_transcript_words: list of words from Whisper's transcript
    # tokenizer: Wav2Vec2 tokenizer
    # This function should robustly merge token segments into word segments.
    # The PyTorch tutorial's `merge_words` function is the reference.
    # This is a highly simplified placeholder.
    word_segments = []
    current_word = ""
    word_start_time = -1

    # This simplified logic assumes tokens directly form words, which is often not true.
    # A proper implementation needs to handle subword units, special tokens, and match against `original_transcript_words`.
    print("[WIP] merge_tokens_to_words_from_alignment: Using simplified placeholder logic.")
    
    # Attempt to reconstruct words and match with original_transcript_words
    # This is still very naive and needs the robustness of the tutorial's approach.
    word_idx = 0
    current_reconstructed_word = ""
    
    if not original_transcript_words:
        return []

    for i, seg in enumerate(token_segments):
        token_str = tokenizer.decode([seg["token_id"]], skip_special_tokens=True).strip()
        
        if not token_str: # Skip empty tokens that might result from special tokens
            continue

        if word_start_time == -1:
            word_start_time = seg["start_time"]

        current_reconstructed_word += token_str
        
        # Try to match the current reconstructed word with the target word
        # This is very basic matching.
        target_word_processed = original_transcript_words[word_idx].lower().replace("'", "")
        reconstructed_processed = current_reconstructed_word.lower().replace("'", "")

        # If current reconstructed word seems to complete the target word, or if it's the last token
        # This condition needs to be much smarter.
        if reconstructed_processed.startswith(target_word_processed) or (i == len(token_segments) -1):
            if reconstructed_processed == target_word_processed or (i == len(token_segments) -1 and reconstructed_processed): # Match or end of tokens
                word_segments.append({
                    "word": original_transcript_words[word_idx],
                    "start_time": word_start_time,
                    "end_time": seg["end_time"]
                })
                word_idx += 1
                current_reconstructed_word = ""
                word_start_time = -1
                if word_idx >= len(original_transcript_words):
                    break 
            # If current reconstructed word is longer than target, something is off, reset for next target word
            elif len(reconstructed_processed) > len(target_word_processed) and not target_word_processed.startswith(reconstructed_processed) :
                 # This indicates a mismatch, try to advance target word if current token seems to start it
                 # Or, if the current token is a standalone part of the next word.
                 # This logic is very tricky. For now, let's just reset and hope next token starts fresh.
                 current_reconstructed_word = token_str # Start new word with current token
                 word_start_time = seg["start_time"]


    # If there are remaining words in transcript that weren't matched (e.g. due to alignment issues)
    # this part is not handled by this simplified logic.
    if not word_segments and original_transcript_words: # Fallback if no words were formed
        print("Wav2Vec2 alignment failed to form words, using basic split for aligned data as fallback.")
        # This fallback is not ideal as it won't have Wav2Vec2 timings.
        # It's better to fall back to Whisper's timings if Wav2Vec2 fails significantly.
        # The main function's structure handles this fallback.
        pass


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
                # Ensure 'word' is not None and strip it
                word_text = segment.get('word', '').strip()
                if word_text: # Only add if word is not empty after stripping
                    detailed_lyrics_data.append({
                        "word": word_text, 
                        "start_time": round(segment['start'], 3),
                        "end_time": round(segment['end'], 3)
                    })
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
            alignment_model_name = cfg.get('wav2vec2_alignment_model', 'facebook/wav2vec2-base-960h')
            alignment_device = cfg.get('alignment_device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            print(f"Loading Wav2Vec2 Processor: {alignment_model_name}...")
            processor = Wav2Vec2Processor.from_pretrained(alignment_model_name)
            print(f"Loading Wav2Vec2 Model: {alignment_model_name} for CTC...")
            model = Wav2Vec2ForCTC.from_pretrained(alignment_model_name).to(alignment_device)

            target_sr_wav2vec2 = processor.feature_extractor.sampling_rate
            waveform_np_resampled = waveform_np_float32
            if sr != target_sr_wav2vec2:
                print(f"Resampling audio from {sr}Hz to {target_sr_wav2vec2}Hz for Wav2Vec2...")
                waveform_np_resampled = librosa.resample(waveform_np_float32, orig_sr=sr, target_sr=target_sr_wav2vec2)
            
            print("Processing audio for Wav2Vec2...")
            input_values = processor(waveform_np_resampled, sampling_rate=target_sr_wav2vec2, return_tensors="pt", padding=True).input_values.to(alignment_device)
            
            print("Getting logits from Wav2Vec2...")
            with torch.no_grad():
                logits = model(input_values).logits.cpu()[0] # Logits for the first batch item

            # The PyTorch tutorial's full alignment logic is complex.
            # The following is a placeholder for that logic.
            # For a real implementation, adapt the tutorial's `get_trellis`, `backtrack`,
            # `merge_repeats`, and `merge_words` (our `merge_tokens_to_words_from_alignment`).
            
            # --- Placeholder for actual alignment logic from PyTorch tutorial ---
            # 1. Prepare transcript tokens for CTC alignment (often with blanks)
            # transcript_for_alignment = "|".join(transcribed_text.upper().split()) # Example, depends on tokenizer
            # tokens_for_alignment = [processor.tokenizer.convert_tokens_to_ids(c) for c in transcript_for_alignment]
            # blank_id = processor.tokenizer.pad_token_id # Or specific blank token ID

            # 2. Compute Trellis & Backtrack (using the robust helper functions)
            # trellis = get_trellis(logits, tokens_for_alignment, blank_id)
            # path = backtrack(trellis, logits, tokens_for_alignment, blank_id) # list of token_ids

            # 3. Calculate time ratio for frames
            # time_ratio = waveform_np_resampled.shape[0] / target_sr_wav2vec2 / logits.size(0)

            # 4. Merge repeated tokens/blanks from path to get token-level segments
            # token_segments = merge_repeats_and_blanks(path, time_ratio, processor.tokenizer, blank_id)
            
            # 5. Merge token segments into word segments
            # refined_word_segments = merge_tokens_to_words_from_alignment(
            #    token_segments,
            #    transcribed_text.split(), # Original words from Whisper
            #    processor.tokenizer
            # )

            # if refined_word_segments:
            #    detailed_lyrics_data = refined_word_segments # Replace Whisper's data
            #    print("Wav2Vec2 alignment applied (partially, with placeholder logic).")
            # else:
            #    print("Wav2Vec2 alignment (placeholder) did not produce refined word segments. Using Whisper's.")
            
            print("[INFO] Wav2Vec2 alignment refinement section uses placeholder logic for CTC segmentation and word merging.")
            print("[INFO] For robust Wav2Vec2 alignment, the helper functions (get_trellis, backtrack, merge_repeats_and_blanks, merge_tokens_to_words_from_alignment) need to be fully implemented based on PyTorch tutorial or similar methods.")
            print("[INFO] Currently, this section does not modify detailed_lyrics_data from Whisper's output.")
            # --- End Placeholder ---

        except Exception as e_wav2vec:
            print(f"Error during Wav2Vec2 alignment setup or placeholder execution: {e_wav2vec}. Using Whisper's timestamps.")
            # In case of error, we continue with the data from Whisper, no need to modify detailed_lyrics_data

    return {
        "text": transcribed_text if transcribed_text else "",
        "detailed_lyrics": detailed_lyrics_data
    }
