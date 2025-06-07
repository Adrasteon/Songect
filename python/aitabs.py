import torchaudio as ta
import torch as th
import torch.nn.functional as F
from librosa.feature import tempo as librosa_feature_tempo # Alias to avoid conflict
import numpy as np
import librosa # Ensure librosa is imported for librosa.beat
import crepe # Added for pitch detection

from .audio import read_wav, write_wav, gen_wav
from .utils import build_masked_stft, get_chord_name, get_segment_name, get_lyrics
from .spec import istft, get_spec, get_specs, get_mixed_spec
from .modulation import search_key
from .models import get_model


class AITabTranscription(object):
    def __init__(self, config):
        self.config = config
        self.n_channel = self.config['n_channel']
        self.sources = self.config['sources']
        self.sample_rate = self.config['sample_rate']
        self.sep_config = self.config['separate']
        self.lyrics_cfg = self.config['lyrics']
        self.beat_cfg = self.config['beat']
        self.chord_cfg = self.config['chord']
        self.segment_cfg = self.config['segment']
        self.pitch_cfg = self.config['pitch']
        self.spec_cfg = self.config['spec']
        self.tempo_cfg = self.config['tempo']

    def separate(self, waveform, sample_rate, device='cpu', unet_model=None):
        assert sample_rate == self.sample_rate
        wav_len = waveform.shape[-1]

        # Use the passed unet_model if available, otherwise load (though it should be passed now)
        if unet_model is None: 
            print("Warning: unet_model not passed to separate method, attempting to load.")
            # This path should ideally not be taken if transcribe method is updated correctly
            unet_model, _ = get_model(self.sep_config['model_name'], self.sep_config['model'], 
                                    model_path=self.sep_config['model_path'], is_train=False, device=device)
        
        # Ensure unet_model is not None before proceeding
        if unet_model is None:
            raise ValueError("UNet model is None in separate method.")

        model_config = self.sep_config['model']
        spec_config = self.sep_config['spec']
        n_fft = self.sep_config['spec']['n_fft']
        hop_length = self.sep_config['spec']['hop_length']
        n_time = self.sep_config['spec']['n_time']

        _model_cfg = {
            'sources': len(self.sources), # Corrected: Pass the number of sources (integer)
            'n_channel': self.n_channel,
        }
        _model_cfg.update(model_config)

        split_len = (n_time - 5) * hop_length + n_fft

        output_waveforms = [[] for _ in range(len(self.sources))] # Corrected: use len(self.sources)
        for i in range(0, wav_len, split_len):
            with th.no_grad():
                x = waveform[:, i:i + split_len]
                pad_num = 0
                if x.shape[-1] < split_len:
                    pad_num = split_len - (wav_len - i)
                    x = F.pad(x, (0, pad_num))

                # separator
                z = get_spec(x, spec_config)
                mag_z = th.abs(z).unsqueeze(0)
                masks = unet_model(mag_z)
                masks = masks.squeeze(0)
                _masked_stfts = build_masked_stft(masks, z, n_fft=n_fft)
                # build waveform
                for j, _masked_stft in enumerate(_masked_stfts):
                    _waveform = istft(_masked_stft, n_fft=n_fft, hop_length=hop_length, pad=True)
                    if pad_num > 0:
                        _waveform = _waveform[:, :-pad_num]
                    output_waveforms[j].append(_waveform)

        inst_waveforms = []
        for waveform_list in output_waveforms:
            inst_waveforms.append(th.cat(waveform_list, dim=-1))
        return th.stack(inst_waveforms, dim=0)

    def transcribe(self, wav_fp, device='cpu'):
        results = {}
        # Flags to track if models were loaded with trained weights
        unet_weights_loaded = False
        # beat_net_weights_loaded = False # BeatNet is currently bypassed for librosa
        chord_net_weights_loaded = False
        segment_net_weights_loaded = False
        # pitch_net is replaced by crepe, which handles its own model loading

        try:
            # Section 1: Audio Reading and Separation
            try:
                waveform, sample_rate = read_wav(wav_fp, sample_rate=self.sample_rate, n_channel=self.n_channel, device=device)
                waveform_np_for_librosa_and_crepe = waveform.cpu().numpy()
                if waveform_np_for_librosa_and_crepe.ndim > 1 and waveform_np_for_librosa_and_crepe.shape[0] == self.n_channel and self.n_channel > 1:
                    waveform_np_for_librosa_and_crepe = np.mean(waveform_np_for_librosa_and_crepe, axis=0)
                elif waveform_np_for_librosa_and_crepe.ndim > 1 and waveform_np_for_librosa_and_crepe.shape[0] == 1:
                    waveform_np_for_librosa_and_crepe = waveform_np_for_librosa_and_crepe.squeeze(0)

                # Prepare config for UNets model
                unet_model_cfg = self.sep_config['model'].copy() # Get base config (conv_n_filters, etc.)
                unet_model_cfg['sources'] = len(self.sources) # Add number of sources
                unet_model_cfg['n_channel'] = self.n_channel    # Add number of input channels

                unet, unet_weights_loaded = get_model(
                    self.sep_config['model_name'], 
                    unet_model_cfg, # Pass the updated config for UNets
                    model_path=self.sep_config['model_path'], 
                    is_train=False, 
                    device=device
                )
                if unet_weights_loaded:
                    inst_waveforms = self.separate(waveform, sample_rate, device=device, unet_model=unet) # Pass loaded unet
                else:
                    # If UNet weights not loaded, create placeholder stems (e.g., mono version of original for all)
                    print("Separation model (UNet) not loaded with weights. Using original audio for stems.")
                    num_sources = len(self.sources)
                    mono_waveform_tensor = th.from_numpy(waveform_np_for_librosa_and_crepe).unsqueeze(0).to(device) 
                    inst_waveforms = th.stack([mono_waveform_tensor.clone() for _ in range(num_sources)], dim=0)
                
                results['stems_debug_shape'] = str(inst_waveforms.shape)

            except TypeError as e:
                print(f"TypeError in Audio Reading/Separation: {e}")
                raise
            except Exception as e:
                print(f"Error in Audio Reading/Separation: {e}")
                raise

            # Section 2: Model Loading (excluding separation model, which is loaded above, and pitch, which is crepe)
            try:
                # BeatNet is currently bypassed for librosa, so no need to load it.
                # beat_net, beat_net_weights_loaded = get_model(self.beat_cfg['model_name'], self.beat_cfg['model'],
                # model_path=self.beat_cfg['model_path'], is_train=False, device=device)
                
                chord_net, chord_net_weights_loaded = get_model(self.chord_cfg['model_name'], self.chord_cfg['model'],
                                      model_path=self.chord_cfg['model_path'], is_train=False, device=device)
                segment_net, segment_net_weights_loaded = get_model(self.segment_cfg['model_name'], self.segment_cfg['model'],
                                        model_path=self.segment_cfg['model_path'], is_train=False, device=device)
                # PitchNet is replaced by crepe
            except TypeError as e:
                print(f"TypeError in Model Loading: {e}")
                raise
            except Exception as e:
                print(f"Error in Model Loading: {e}")
                raise

            # Section 3: Spectrogram Generation
            try:
                vocal_waveform_np = inst_waveforms[0].cpu().numpy() # For get_lyrics
                orig_spec = get_spec(waveform, self.spec_cfg)
                inst_specs = get_specs(inst_waveforms, self.spec_cfg)  # vocal, bass, drum, other
                vocal_spec = get_spec(inst_waveforms[0], self.spec_cfg)  # vocal
                other_spec = get_mixed_spec(inst_waveforms[1:], self.spec_cfg)  # bass + drum + other
            except TypeError as e:
                print(f"TypeError in Spectrogram Generation: {e}")
                raise
            except Exception as e:
                print(f"Error in Spectrogram Generation: {e}")
                raise

            # Section 4: Lyrics Prediction
            lyrics_data_dict = {} # Initialize
            try:
                # vocal_waveform_np for get_lyrics should be from the separated vocals if separation worked,
                # otherwise, it's from the mono mix.
                if unet_weights_loaded and inst_waveforms.shape[0] > 0: # Check if separation produced stems
                    vocal_waveform_np = inst_waveforms[self.sources.index("vocals") if "vocals" in self.sources else 0].cpu().numpy()
                    if vocal_waveform_np.ndim > 1 and vocal_waveform_np.shape[0] == 1: # Ensure it's 1D for get_lyrics
                        vocal_waveform_np = vocal_waveform_np.squeeze(0)
                else:
                    vocal_waveform_np = waveform_np_for_librosa_and_crepe # Use original mono mix
                
                lyrics_data_dict = get_lyrics(vocal_waveform_np, sample_rate, self.lyrics_cfg)
            except TypeError as e:
                print(f"TypeError in Lyrics Prediction (get_lyrics): {e}")
                raise
            except Exception as e:
                print(f"Error in Lyrics Prediction (get_lyrics): {e}")
                lyrics_data_dict = {"text": "Lyrics error", "detailed_lyrics": []} # Fallback

            with th.no_grad():
                # Section 5: Beat Prediction (using librosa)
                beats_result_list = []
                downbeats_result_list = []
                bpm_result = 0.0
                try:
                    bpm_val, beat_times_val, downbeat_times_val = self.get_beat_librosa(waveform_np_for_librosa_and_crepe, sample_rate)
                    beats_result_list = beat_times_val.tolist() if isinstance(beat_times_val, np.ndarray) else beat_times_val
                    downbeats_result_list = downbeat_times_val.tolist() if isinstance(downbeat_times_val, np.ndarray) else downbeat_times_val
                    bpm_result = bpm_val
                    print(f"Librosa: BPM={bpm_result}, Beats found: {len(beats_result_list)}")
                except Exception as e:
                    print(f"Error in Librosa Beat/Tempo Prediction: {e}")
                    beats_result_list = ["Beat/BPM: Librosa Error / WIP"]
                    bpm_result = "WIP"

                # Section 6: Chord Prediction
                chords_result = []
                if chord_net_weights_loaded:
                    try:
                        chord_features_mag = other_spec.permute(0, 2, 1).unsqueeze(0)
                        chord_pred_output, _ = chord_net(chord_features_mag, y=None)
                        chord_indices_np = chord_pred_output.squeeze(0).squeeze(0).cpu().numpy()
                        if not isinstance(chord_indices_np, np.ndarray):
                            raise TypeError(f"chord_pred_output.cpu().numpy() did not return np.ndarray, got {type(chord_indices_np)}. Value: {chord_indices_np}")
                        chord_indices = chord_indices_np.flatten()
                        chords_result = get_chord_name(chord_indices)
                    except TypeError as e:
                        print(f"TypeError in Chord Prediction: {e}")
                        chords_result = ["Chords: TypeError / WIP"]
                    except Exception as e:
                        print(f"Error in Chord Prediction: {e}")
                        chords_result = ["Chords: Error / WIP"]
                else:
                    chords_result = ["Chords: Model Not Loaded / WIP"]

                # Section 7: Segment Prediction
                segments_result = []
                if segment_net_weights_loaded:
                    try:
                        segment_features_mag = orig_spec.permute(0, 2, 1).unsqueeze(0)
                        segment_pred_output, _ = segment_net(segment_features_mag, y=None) 
                        segments_idx_np = segment_pred_output.squeeze(0).squeeze(0).cpu().numpy()
                        if not isinstance(segments_idx_np, np.ndarray):
                            raise TypeError(f"segment_pred_output.cpu().numpy() did not return np.ndarray, got {type(segments_idx_np)}. Value: {segments_idx_np}")
                        segments_idx = segments_idx_np.flatten()
                        segments_result = get_segment_name(segments_idx)
                    except TypeError as e:
                        print(f"TypeError in Segment Prediction: {e}")
                        segments_result = ["Segments: TypeError / WIP"]
                    except Exception as e:
                        print(f"Error in Segment Prediction: {e}")
                        segments_result = ["Segments: Error / WIP"]
                else:
                    segments_result = ["Segments: Model Not Loaded / WIP"]

                # Section 8: Key Prediction
                key_result = "N/A"
                if chord_net_weights_loaded and chords_result and "WIP" not in chords_result[0]:
                    try:
                        key_result = search_key(chords_result) 
                    except TypeError as e:
                        print(f"TypeError in Key Prediction: {e}")
                        key_result = "Key: TypeError / WIP"
                    except Exception as e:
                        print(f"Error in Key Prediction: {e}")
                        key_result = "Key: Error / WIP"
                else:
                    key_result = "Key: Chord Model Not Loaded or Error / WIP"

                # Section 9: Pitch Prediction (using CREPE)
                pitch_list = []
                try:
                    # Ensure vocal_waveform_np_for_crepe is 1D float32 and 16kHz for CREPE
                    vocal_waveform_np_for_crepe = waveform_np_for_librosa_and_crepe # This is already mono
                    if sample_rate != 16000:
                        print(f"Resampling audio from {sample_rate}Hz to 16000Hz for CREPE...")
                        vocal_waveform_np_for_crepe = librosa.resample(vocal_waveform_np_for_crepe.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
                    
                    # CREPE expects audio between -1 and 1
                    if np.max(np.abs(vocal_waveform_np_for_crepe)) > 1.0:
                        vocal_waveform_np_for_crepe = vocal_waveform_np_for_crepe / np.max(np.abs(vocal_waveform_np_for_crepe))

                    time, frequency, confidence, activation = crepe.predict(vocal_waveform_np_for_crepe, 16000, model_capacity="full", step_size=10, verbose=0)
                    # For pitch_list, we can store (time, frequency, confidence) tuples or just frequencies
                    # Taking frequencies where confidence is high, e.g., > 0.5
                    # pitch_list = frequency[confidence > 0.5].tolist()
                    # Or, for now, just return all frequencies to match previous structure if it was just a list of floats
                    pitch_list = frequency.tolist() 
                    print(f"CREPE pitch detection completed. Found {len(pitch_list)} pitch points.")
                except ImportError:
                    print("CREPE library not found. Skipping pitch detection.")
                    pitch_list = ["Pitch: CREPE Not Found / WIP"]
                except Exception as e:
                    print(f"Error in CREPE Pitch Prediction: {e}")
                    pitch_list = ["Pitch: CREPE Error / WIP"]

            # Section 10: Format Results
            stems_np_dict = {}
            if unet_weights_loaded:
                try:
                    stems_np_dict = {
                        source_name: inst_waveforms[i].cpu().numpy() 
                        for i, source_name in enumerate(self.sources)
                        if i < inst_waveforms.shape[0] # Ensure index is within bounds
                    }
                except Exception as e:
                    print(f"Error formatting stems: {e}")
                    stems_np_dict = {"stems_error": "Error processing stems / WIP"}
            else:
                stems_np_dict = {"stems_status": "Separation Model Not Loaded / WIP"}

            results = {
                'lyrics': lyrics_data_dict, 
                'beats': beats_result_list,
                'downbeats': downbeats_result_list,
                'bpm': bpm_result,
                'chords': chords_result,
                'key': key_result,
                'segments': segments_result,
                'pitch': pitch_list,
                'stems': stems_np_dict
            }
            return results

        except Exception as e: # Catch-all for any other unexpected error in transcribe
            print(f"Overall error in AITabTranscription.transcribe for {wav_fp}: {e}")
            # Return a dictionary with an error field, or re-raise, or return None
            # For consistency with how BackendHandler might expect None on error:
            # raise # Or return a dict with an error message
            return {
                'error': str(e),
                'lyrics': {"text": "Transcription error", "detailed_lyrics": []},
                'beats': [], 'downbeats': [], 'bpm': 0.0, 'chords': [], 'key': 'N/A',
                'segments': [], 'pitch': [], 'stems': {}
            }

    def get_beat_librosa(self, waveform_np, sample_rate):
        """
        Calculates BPM and beat times using librosa.
        Args:
            waveform_np (np.ndarray): Input waveform (mono, float32).
            sample_rate (int): Sample rate of the waveform.
        Returns:
            tuple: (bpm, beat_times, downbeat_times)
                   bpm (float): Estimated BPM.
                   beat_times (np.ndarray): Array of beat timestamps in seconds.
                   downbeat_times (np.ndarray): Array of downbeat timestamps in seconds (currently empty).
        """
        bpm = 0.0
        beat_times = np.array([])
        downbeat_times = np.array([]) # Placeholder for now

        if waveform_np is None or len(waveform_np) == 0:
            print("Warning: get_beat_librosa received empty or None waveform.")
            return bpm, beat_times, downbeat_times

        try:
            # Ensure waveform_np is float32, as librosa expects
            waveform_np = waveform_np.astype(np.float32)
            if waveform_np.ndim > 1:
                waveform_np = np.mean(waveform_np, axis=0)

            # Calculate tempo (BPM)
            # Updated to use librosa.feature.rhythm.tempo to address FutureWarning
            tempo_values = librosa.feature.rhythm.tempo(y=waveform_np, sr=sample_rate, hop_length=self.tempo_cfg.get('hop_length', 512))
            bpm = tempo_values[0] if len(tempo_values) > 0 else 0.0

            # Calculate beat frames
            # beat_track returns estimated tempo and beat frames
            _, beat_frames = librosa.beat.beat_track(y=waveform_np, sr=sample_rate, hop_length=self.tempo_cfg.get('hop_length', 512))
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=self.tempo_cfg.get('hop_length', 512))
            
            # Downbeat tracking with librosa is more involved and often less direct than beat tracking.
            # For now, we return an empty array for downbeats.
            # True downbeat detection might require a dedicated model or more complex logic.
            print(f"Librosa beat tracking: BPM={bpm}, Found {len(beat_times)} beats.")

        except Exception as e:
            print(f"Error in librosa beat/tempo processing: {e}")
            bpm = 0.0
            beat_times = np.array([])
            # downbeat_times remains empty
        
        return float(bpm), beat_times, downbeat_times

