
#with_fx :reverb, mix: 0.7 do


live_loop :note0 do
  use_synth :hollow
  play_pattern [:C2,:C3].ring
  sleep 0.5
end

live_loop :note1 do
  use_synth :bass_foundation
  use_octave -2
  a, b = sync "/osc*/trigger2/prophet"
  play choose([a,b]), attack: 6, release: 6
  sleep 0.5
end


live_loop :note2 do
  use_synth :hollow
  a, b = sync "/osc*/trigger2/prophet"
  play a
  sleep 0.5
end

live_loop :note3 do
  use_synth :hollow
  a, b = sync "/osc*/trigger2/prophet"
  play b
  sleep 0.5
end

live_loop :tijd do
  sample :bd_haus, amp: 0.5, cutoff: 100
  sleep 0.5
end
