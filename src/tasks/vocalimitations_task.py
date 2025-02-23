from xares.task import TaskConfig


def vocalimiations_config(encoder) -> TaskConfig:
    task = "imitation"
    class_label_maps = {
        "000Animal_Domestic animals_ pets_Cat_Growling_reference": 0,
        "001Animal_Domestic animals_ pets_Cat_Hiss_reference": 1,
        "002Animal_Domestic animals_ pets_Cat_Meow_reference": 2,
        "003Animal_Domestic animals_ pets_Cat_Purr_reference": 3,
        "004Animal_Domestic animals_ pets_Dog_Bark_reference": 4,
        "005Animal_Domestic animals_ pets_Dog_Growling_reference": 5,
        "006Animal_Domestic animals_ pets_Dog_Howl_reference": 6,
        "007Animal_Livestock_ farm animals_ working animals_Cattle_ bovinae_Cowbell_reference": 7,
        "008Animal_Livestock_ farm animals_ working animals_Cattle_ bovinae_Moo_reference": 8,
        "009Animal_Livestock_ farm animals_ working animals_Fowl_Chicken_ rooster_Cluck_reference": 9,
        "010Animal_Livestock_ farm animals_ working animals_Fowl_Chicken_ rooster_Crowing_ cock-a-doodle-doo_reference": 10,
        "011Animal_Livestock_ farm animals_ working animals_Fowl_Duck_Quack_reference": 11,
        "012Animal_Livestock_ farm animals_ working animals_Fowl_Goose_Honk_reference": 12,
        "013Animal_Livestock_ farm animals_ working animals_Fowl_Turkey_Gobble_reference": 13,
        "014Animal_Livestock_ farm animals_ working animals_Goat_Bleat_reference": 14,
        "015Animal_Livestock_ farm animals_ working animals_Horse_Clip-clop_reference": 15,
        "016Animal_Livestock_ farm animals_ working animals_Horse_Neigh_ whinny_reference": 16,
        "017Animal_Livestock_ farm animals_ working animals_Horse_Snort (horse)_reference": 17,
        "018Animal_Livestock_ farm animals_ working animals_Pig_Oink_reference": 18,
        "019Animal_Wild animals_Bird_Bird flight_ flapping wings_reference": 19,
        "020Animal_Wild animals_Bird_Bird vocalization_ bird call_ bird song_Chirp_ tweet_reference": 20,
        "021Animal_Wild animals_Bird_Crow_Caw_reference": 21,
        "022Animal_Wild animals_Bird_Owl_Hoot_reference": 22,
        "023Animal_Wild animals_Bird_Pigeon_ dove_Coo_reference": 23,
        "024Animal_Wild animals_Frog_Croak_reference": 24,
        "025Animal_Wild animals_Insect_Bee_ wasp_ etc_Buzz_reference": 25,
        "026Animal_Wild animals_Insect_Cricket_reference": 26,
        "027Animal_Wild animals_Insect_Mosquito_reference": 27,
        "028Animal_Wild animals_Roaring cats (lions_ tigers)_Roar_reference": 28,
        "029Animal_Wild animals_Rodents_ rats_ mice_Mouse___3Samples_reference": 29,
        "030Animal_Wild animals_Snake_Rattle_reference": 30,
        "031Channel_ environment and background_Noise_Background noise_Mains hum_reference": 31,
        "032Channel_ environment and background_Noise_Background noise_Tape hiss_reference": 32,
        "033Channel_ environment and background_Noise_Pink noise_reference": 33,
        "034Channel_ environment and background_Noise_White noise_reference": 34,
        "035Human sounds_Digestive_Biting_ Chewing_reference": 35,
        "036Human sounds_Digestive_Burping_ eructation_reference": 36,
        "037Human sounds_Digestive_Fart_reference": 37,
        "038Human sounds_Digestive_Gargling_reference": 38,
        "039Human sounds_Digestive_Hiccup_reference": 39,
        "040Human sounds_Digestive_Stomach rumble_reference": 40,
        "041Human sounds_Hands_Clapping_reference": 41,
        "042Human sounds_Hands_Finger snapping_reference": 42,
        "043Human sounds_Heart sounds_ heartbeat_reference": 43,
        "044Human sounds_Human group actions_Applause_reference": 44,
        "045Human sounds_Human group actions_Booing_reference": 45,
        "046Human sounds_Human group actions_Cheering_reference": 46,
        "047Human sounds_Human group actions_Children shouting_reference": 47,
        "048Human sounds_Human locomotion_Run_reference": 48,
        "049Human sounds_Human locomotion_Shuffle_reference": 49,
        "050Human sounds_Human locomotion_Walk_ footsteps_reference": 50,
        "051Human sounds_Human voice_Crying_ sobbing_Baby cry_ infant cry_reference": 51,
        "052Human sounds_Human voice_Groan_reference": 52,
        "053Human sounds_Human voice_Grunt_reference": 53,
        "054Human sounds_Human voice_Humming_reference": 54,
        "055Human sounds_Human voice_Laughter_Baby laughter_reference": 55,
        "056Human sounds_Human voice_Laughter_Chuckle_ chortle_reference": 56,
        "057Human sounds_Human voice_Laughter_Giggle_reference": 57,
        "058Human sounds_Human voice_Screaming_reference": 58,
        "059Human sounds_Human voice_Shout_Children shouting_reference": 59,
        "060Human sounds_Human voice_Shout_Whoop_reference": 60,
        "061Human sounds_Human voice_Shout_Yell_reference": 61,
        "062Human sounds_Human voice_Sigh_reference": 62,
        "063Human sounds_Human voice_Wail_ moan_reference": 63,
        "064Human sounds_Human voice_Whispering_reference": 64,
        "065Human sounds_Human voice_Yawn_reference": 65,
        "066Human sounds_Respiratory sounds_Breathing_Gasp_reference": 66,
        "067Human sounds_Respiratory sounds_Breathing_Pant_reference": 67,
        "068Human sounds_Respiratory sounds_Breathing_Snoring_reference": 68,
        "069Human sounds_Respiratory sounds_Cough_Throat clearing_reference": 69,
        "070Human sounds_Respiratory sounds_Sneeze_reference": 70,
        "071Human sounds_Respiratory sounds_Sniff_reference": 71,
        "072Human sounds_Whistling_Wolf-whistling_reference": 72,
        "073Music_Musical instrument_Accordion_reference": 73,
        "074Music_Musical instrument_Bagpipes_reference": 74,
        "075Music_Musical instrument_Bell_Bicycle bell_reference": 75,
        "076Music_Musical instrument_Bell_Chime_Wind chime_reference": 76,
        "077Music_Musical instrument_Bell_Church bell_reference": 77,
        "078Music_Musical instrument_Bell_Jingle bell_reference": 78,
        "079Music_Musical instrument_Bell_Tuning fork_reference": 79,
        "080Music_Musical instrument_Bowed string instrument_Cello_reference": 80,
        "081Music_Musical instrument_Bowed string instrument_Double bass_reference": 81,
        "082Music_Musical instrument_Bowed string instrument_String section_reference": 82,
        "083Music_Musical instrument_Bowed string instrument_Violin_ fiddle_Pizzicato_reference": 83,
        "084Music_Musical instrument_Brass instrument_Bugle_reference": 84,
        "085Music_Musical instrument_Brass instrument_Cornet_reference": 85,
        "086Music_Musical instrument_Brass instrument_French horn_reference": 86,
        "087Music_Musical instrument_Brass instrument_Trombone_reference": 87,
        "088Music_Musical instrument_Brass instrument_Trumpet_reference": 88,
        "089Music_Musical instrument_Choir_reference": 89,
        "090Music_Musical instrument_Didgeridoo_reference": 90,
        "091Music_Musical instrument_Harmonica_reference": 91,
        "092Music_Musical instrument_Harp_reference": 92,
        "093Music_Musical instrument_Keyboard (musical)_Harpsichord_reference": 93,
        "094Music_Musical instrument_Keyboard (musical)_Organ_Electronic organ_reference": 94,
        "095Music_Musical instrument_Keyboard (musical)_Organ_Hammond organ_reference": 95,
        "096Music_Musical instrument_Keyboard (musical)_Piano_Electric piano_reference": 96,
        "097Music_Musical instrument_Keyboard (musical)_Synthesizer_Mellotron_reference": 97,
        "098Music_Musical instrument_Keyboard (musical)_Synthesizer_Sampler_reference": 98,
        "099Music_Musical instrument_Musical ensemble_reference": 99,
        "100Music_Musical instrument_Orchestra_reference": 100,
        "101Music_Musical instrument_Percussion_Cowbell_reference": 101,
        "102Music_Musical instrument_Percussion_Cymbal_Crash Cymbal_reference": 102,
        "103Music_Musical instrument_Percussion_Cymbal_Hi-hat_reference": 103,
        "104Music_Musical instrument_Percussion_Drum kit_Drum machine_reference": 104,
        "105Music_Musical instrument_Percussion_Drum_Bass drum_reference": 105,
        "106Music_Musical instrument_Percussion_Drum_Snare drum_Drum roll_reference": 106,
        "107Music_Musical instrument_Percussion_Drum_Snare drum_Rimshot_reference": 107,
        "108Music_Musical instrument_Percussion_Drum_Tabla_reference": 108,
        "109Music_Musical instrument_Percussion_Drum_Timpani_reference": 109,
        "110Music_Musical instrument_Percussion_Gong_reference": 110,
        "111Music_Musical instrument_Percussion_Mallet percussion_Glockenspiel_reference": 111,
        "112Music_Musical instrument_Percussion_Mallet percussion_Marimba_ xylophone_reference": 112,
        "113Music_Musical instrument_Percussion_Mallet percussion_Vibraphone_reference": 113,
        "114Music_Musical instrument_Percussion_Rattle (instrument)_Maraca_reference": 114,
        "115Music_Musical instrument_Percussion_Tambourine_reference": 115,
        "116Music_Musical instrument_Percussion_Tubular bells_reference": 116,
        "117Music_Musical instrument_Percussion_Wood block_reference": 117,
        "118Music_Musical instrument_Plucked string instrument_Banjo_reference": 118,
        "119Music_Musical instrument_Plucked string instrument_Guitar_Acoustic guitar_reference": 119,
        "120Music_Musical instrument_Plucked string instrument_Guitar_Bass guitar_reference": 120,
        "121Music_Musical instrument_Plucked string instrument_Guitar_Electric guitar_reference": 121,
        "122Music_Musical instrument_Plucked string instrument_Guitar_Steel guitar_ slide guitar_reference": 122,
        "123Music_Musical instrument_Plucked string instrument_Guitar_Strum_reference": 123,
        "124Music_Musical instrument_Plucked string instrument_Guitar_Tapping (guitar technique)_reference": 124,
        "125Music_Musical instrument_Plucked string instrument_Mandolin_reference": 125,
        "126Music_Musical instrument_Plucked string instrument_Sitar_reference": 126,
        "127Music_Musical instrument_Plucked string instrument_Ukulele_reference": 127,
        "128Music_Musical instrument_Plucked string instrument_Zither_reference": 128,
        "129Music_Musical instrument_Scratching (performance technique)_reference": 129,
        "130Music_Musical instrument_Singing bowl_reference": 130,
        "131Music_Musical instrument_Theremin_reference": 131,
        "132Music_Musical instrument_Wind instrument_ woodwind instrument_Bassoon_reference": 132,
        "133Music_Musical instrument_Wind instrument_ woodwind instrument_Clarinet_reference": 133,
        "134Music_Musical instrument_Wind instrument_ woodwind instrument_Flute_reference": 134,
        "135Music_Musical instrument_Wind instrument_ woodwind instrument_Oboe_reference": 135,
        "136Music_Musical instrument_Wind instrument_ woodwind instrument_Saxophone_Alto saxophone_reference": 136,
        "137Music_Musical instrument_Wind instrument_ woodwind instrument_Saxophone_Soprano saxophone_reference": 137,
        "138Natural sounds_Fire_Crackle_reference": 138,
        "139Natural sounds_Thunderstorm_Thunder_reference": 139,
        "140Natural sounds_Water_Gurgling_reference": 140,
        "141Natural sounds_Water_Ocean_Waves_ surf_reference": 141,
        "142Natural sounds_Water_Rain_Rain on surface_reference": 142,
        "143Natural sounds_Water_Stream_reference": 143,
        "144Natural sounds_Water_Steam_Hiss_reference": 144,
        "145Natural sounds_Water_Waterfall_reference": 145,
        "146Natural sounds_Wind_Howl (wind)_reference": 146,
        "147Natural sounds_Wind_Rustling leaves_reference": 147,
        "148Sounds of things_Alarm_Air horn_ truck horn_reference": 148,
        "149Sounds of things_Alarm_Alarm clock_reference": 149,
        "150Sounds of things_Alarm_Bicycle bell_reference": 150,
        "151Sounds of things_Alarm_Buzzer_reference": 151,
        "152Sounds of things_Alarm_Car alarm_reference": 152,
        "153Sounds of things_Alarm_Doorbell_Ding-dong_reference": 153,
        "154Sounds of things_Alarm_Fire alarm_reference": 154,
        "155Sounds of things_Alarm_Foghorn_reference": 155,
        "156Sounds of things_Alarm_Siren_Ambulance (siren)_reference": 156,
        "157Sounds of things_Alarm_Siren_Civil defense siren_reference": 157,
        "158Sounds of things_Alarm_Siren_Fire engine_ fire truck (siren)_reference": 158,
        "159Sounds of things_Alarm_Siren_Police car (siren)_reference": 159,
        "160Sounds of things_Alarm_Telephone_Busy signal_reference": 160,
        "161Sounds of things_Alarm_Telephone_Cellphone buzz_ vibrating alert_reference": 161,
        "162Sounds of things_Alarm_Telephone_Dial tone_reference": 162,
        "163Sounds of things_Alarm_Telephone_Ringtone_reference": 163,
        "164Sounds of things_Alarm_Telephone_Telephone bell ringing_reference": 164,
        "165Sounds of things_Alarm_Telephone_Telephone dialing_ DTMF_reference": 165,
        "166Sounds of things_Alarm_Vehicle horn_ car horn_ honking_Toot_reference": 166,
        "167Sounds of things_Alarm_Whistle_Kettle whistle_reference": 167,
        "168Sounds of things_Bell_Chime_Wind chime_reference": 168,
        "169Sounds of things_Domestic sounds_ home sounds_Bathtub (filling or washing)_reference": 169,
        "170Sounds of things_Domestic sounds_ home sounds_Blender_reference": 170,
        "171Sounds of things_Domestic sounds_ home sounds_Chopping (food)_reference": 171,
        "172Sounds of things_Domestic sounds_ home sounds_Coin (dropping)_reference": 172,
        "173Sounds of things_Domestic sounds_ home sounds_Cupboard open or close_reference": 173,
        "174Sounds of things_Domestic sounds_ home sounds_Cutlery_ silverware_reference": 174,
        "175Sounds of things_Domestic sounds_ home sounds_Dishes_ pots_ and pans_reference": 175,
        "176Sounds of things_Domestic sounds_ home sounds_Door_Knock_reference": 176,
        "177Sounds of things_Domestic sounds_ home sounds_Door_Slam_reference": 177,
        "178Sounds of things_Domestic sounds_ home sounds_Door_Sliding door_reference": 178,
        "179Sounds of things_Domestic sounds_ home sounds_Door_Squeak_reference": 179,
        "180Sounds of things_Domestic sounds_ home sounds_Drawer open or close_reference": 180,
        "181Sounds of things_Domestic sounds_ home sounds_Electric shaver_ electric razor_reference": 181,
        "182Sounds of things_Domestic sounds_ home sounds_Frying (food)_reference": 182,
        "183Sounds of things_Domestic sounds_ home sounds_Hair dryer_reference": 183,
        "184Sounds of things_Domestic sounds_ home sounds_Keys jangling_reference": 184,
        "185Sounds of things_Domestic sounds_ home sounds_Microwave oven_reference": 185,
        "186Sounds of things_Domestic sounds_ home sounds_Packing tape_ duct tape_reference": 186,
        "187Sounds of things_Domestic sounds_ home sounds_Scissors_reference": 187,
        "188Sounds of things_Domestic sounds_ home sounds_Shuffling cards_reference": 188,
        "189Sounds of things_Domestic sounds_ home sounds_Sink (filling or washing)_reference": 189,
        "190Sounds of things_Domestic sounds_ home sounds_Toilet flush_reference": 190,
        "191Sounds of things_Domestic sounds_ home sounds_Toothbrush_Electric toothbrush_reference": 191,
        "192Sounds of things_Domestic sounds_ home sounds_Typing_Computer keyboard_reference": 192,
        "193Sounds of things_Domestic sounds_ home sounds_Typing_Typewriter_reference": 193,
        "194Sounds of things_Domestic sounds_ home sounds_Vacuum cleaner_reference": 194,
        "195Sounds of things_Domestic sounds_ home sounds_Velcro_ hook and loop fastener_reference": 195,
        "196Sounds of things_Domestic sounds_ home sounds_Water tap_ faucet_reference": 196,
        "197Sounds of things_Domestic sounds_ home sounds_Writing_reference": 197,
        "198Sounds of things_Domestic sounds_ home sounds_Zipper (clothing)_reference": 198,
        "199Sounds of things_Engine_Accelerating_ revving_ vroom_reference": 199,
        "200Sounds of things_Engine_Engine starting_reference": 200,
        "201Sounds of things_Engine_Idling_reference": 201,
        "202Sounds of things_Engine_Jet engine_reference": 202,
        "203Sounds of things_Engine_Light engine (high frequency)_Chainsaw_reference": 203,
        "204Sounds of things_Engine_Light engine (high frequency)_Dental drill_ dentist_s drill_reference": 204,
        "205Sounds of things_Engine_Light engine (high frequency)_Lawn mower_reference": 205,
        "206Sounds of things_Engine_Medium engine (mid frequency)_reference": 206,
        "207Sounds of things_Explosion_Boom_Sonic boom_reference": 207,
        "208Sounds of things_Explosion_Burst_ pop_reference": 208,
        "209Sounds of things_Explosion_Eruption_reference": 209,
        "210Sounds of things_Explosion_Fireworks_Firecracker_reference": 210,
        "211Sounds of things_Explosion_Gunshot_ gunfire_Artillery fire_reference": 211,
        "212Sounds of things_Explosion_Gunshot_ gunfire_Cap gun_reference": 212,
        "213Sounds of things_Explosion_Gunshot_ gunfire_Fusillade_reference": 213,
        "214Sounds of things_Explosion_Gunshot_ gunfire_Machine gun_reference": 214,
        "215Sounds of things_Glass_Chink_ clink_reference": 215,
        "216Sounds of things_Glass_Shatter_reference": 216,
        "217Sounds of things_Liquid_Boiling_reference": 217,
        "218Sounds of things_Liquid_Drip_reference": 218,
        "219Sounds of things_Liquid_Fill (with liquid)_reference": 219,
        "220Sounds of things_Liquid_Pour_Gush_reference": 220,
        "221Sounds of things_Liquid_Pour_Trickle_ dribble_reference": 221,
        "222Sounds of things_Liquid_Pump (liquid)_reference": 222,
        "223Sounds of things_Liquid_Splash_ splatter_Slosh_reference": 223,
        "224Sounds of things_Liquid_Spray_reference": 224,
        "225Sounds of things_Liquid_Squish_reference": 225,
        "226Sounds of things_Liquid_Stir_reference": 226,
        "227Sounds of things_Mechanisms_Air conditioning_reference": 227,
        "228Sounds of things_Mechanisms_Camera_Single-lens reflex camera_reference": 228,
        "229Sounds of things_Mechanisms_Cash register_reference": 229,
        "230Sounds of things_Mechanisms_Clock_Tick_reference": 230,
        "231Sounds of things_Mechanisms_Clock_Tick-tock_reference": 231,
        "232Sounds of things_Mechanisms_Gears_reference": 232,
        "233Sounds of things_Mechanisms_Mechanical fan_reference": 233,
        "234Sounds of things_Mechanisms_Printer_reference": 234,
        "235Sounds of things_Mechanisms_Pulleys_reference": 235,
        "236Sounds of things_Mechanisms_Ratchet_ pawl_reference": 236,
        "237Sounds of things_Mechanisms_Sewing machine_reference": 237,
        "238Sounds of things_Miscellaneous sources_Arrow_Thump_ thud_Clunk_reference": 238,
        "239Sounds of things_Miscellaneous sources_Duck call (hunting tool)_reference": 239,
        "240Sounds of things_Miscellaneous sources_Sonar_reference": 240,
        "241Sounds of things_Specific impact sounds_Basketball bounce_reference": 241,
        "242Sounds of things_Tools_Filing (rasp)_reference": 242,
        "243Sounds of things_Tools_Hammer_reference": 243,
        "244Sounds of things_Tools_Jackhammer_reference": 244,
        "245Sounds of things_Tools_Power tool_Drill_Dental drill_ dentist_s drill_reference": 245,
        "246Sounds of things_Tools_Sanding_reference": 246,
        "247Sounds of things_Tools_Sawing_reference": 247,
        "248Sounds of things_Vehicle_Aircraft_Aircraft engine_Propeller_ airscrew_reference": 248,
        "249Sounds of things_Vehicle_Aircraft_Fixed-wing aircraft_ airplane_reference": 249,
        "250Sounds of things_Vehicle_Aircraft_Helicopter_reference": 250,
        "251Sounds of things_Vehicle_Boat_ Water vehicle_Motorboat_ speedboat_reference": 251,
        "252Sounds of things_Vehicle_Boat_ Water vehicle_Rowboat_ canoe_ kayak_reference": 252,
        "253Sounds of things_Vehicle_Boat_ Water vehicle_Sailboat_ sailing ship_reference": 253,
        "254Sounds of things_Vehicle_Boat_ Water vehicle_Ship_reference": 254,
        "255Sounds of things_Vehicle_Motor vehicle (road)_Bus_reference": 255,
        "256Sounds of things_Vehicle_Motor vehicle (road)_Car_Car alarm_reference": 256,
        "257Sounds of things_Vehicle_Motor vehicle (road)_Car_Car passing by_reference": 257,
        "258Sounds of things_Vehicle_Motor vehicle (road)_Car_Power windows_ electric windows_reference": 258,
        "259Sounds of things_Vehicle_Motor vehicle (road)_Car_Race car_ auto racing_reference": 259,
        "260Sounds of things_Vehicle_Motor vehicle (road)_Car_Skidding_reference": 260,
        "261Sounds of things_Vehicle_Motor vehicle (road)_Car_Vehicle horn_ car horn_ honking_Toot_reference": 261,
        "262Sounds of things_Vehicle_Motor vehicle (road)_Emergency vehicle_Ambulance (siren)_reference": 262,
        "263Sounds of things_Vehicle_Motor vehicle (road)_Emergency vehicle_Fire engine_ fire truck (siren)_reference": 263,
        "264Sounds of things_Vehicle_Motor vehicle (road)_Emergency vehicle_Police car (siren)_reference": 264,
        "265Sounds of things_Vehicle_Motor vehicle (road)_Motorcycle_reference": 265,
        "266Sounds of things_Vehicle_Motor vehicle (road)_Traffic noise_ roadway noise_reference": 266,
        "267Sounds of things_Vehicle_Motor vehicle (road)_Truck_Air brake_reference": 267,
        "268Sounds of things_Vehicle_Motor vehicle (road)_Truck_Air horn_ truck horn_reference": 268,
        "269Sounds of things_Vehicle_Motor vehicle (road)_Truck_Ice cream truck_ ice cream van_reference": 269,
        "270Sounds of things_Vehicle_Motor vehicle (road)_Truck_Reversing beeps_reference": 270,
        "271Sounds of things_Vehicle_Non-motorized land vehicle_Bicycle_Bicycle bell_reference": 271,
        "272Sounds of things_Vehicle_Non-motorized land vehicle_Skateboard_reference": 272,
        "273Sounds of things_Vehicle_Rail transport_Railroad car_ train wagon_reference": 273,
        "274Sounds of things_Vehicle_Rail transport_Subway_ metro_ underground_reference": 274,
        "275Sounds of things_Vehicle_Rail transport_Train wheels squealing_reference": 275,
        "276Sounds of things_Vehicle_Rail transport_Train_Train horn_reference": 276,
        "277Sounds of things_Vehicle_Rail transport_Train_Train whistle_reference": 277,
        "278Sounds of things_Wood_Chop_reference": 278,
        "279Sounds of things_Wood_Crack_reference": 279,
        "280Sounds of things_Wood_Snap_reference": 280,
        "281Sounds of things_Wood_Splinter_reference": 281,
        "282Source-ambiguous sounds_Deformable shell_Crumpling_ crinkling_reference": 282,
        "283Source-ambiguous sounds_Deformable shell_Crushing_reference": 283,
        "284Source-ambiguous sounds_Deformable shell_Tearing_reference": 284,
        "285Source-ambiguous sounds_Generic impact sounds_Bang_reference": 285,
        "286Source-ambiguous sounds_Generic impact sounds_Bouncing_reference": 286,
        "287Source-ambiguous sounds_Generic impact sounds_Breaking_reference": 287,
        "288Source-ambiguous sounds_Generic impact sounds_Flap_reference": 288,
        "289Source-ambiguous sounds_Generic impact sounds_Knock_reference": 289,
        "290Source-ambiguous sounds_Generic impact sounds_Slap_ smack_reference": 290,
        "291Source-ambiguous sounds_Generic impact sounds_Smash_ crash_reference": 291,
        "292Source-ambiguous sounds_Generic impact sounds_Tap_reference": 292,
        "293Source-ambiguous sounds_Generic impact sounds_Thump_ thud_Clunk_reference": 293,
        "294Source-ambiguous sounds_Generic impact sounds_Thump_ thud_Thunk_reference": 294,
        "295Source-ambiguous sounds_Generic impact sounds_Whack_ thwack_reference": 295,
        "296Source-ambiguous sounds_Generic impact sounds_Whip_reference": 296,
        "297Source-ambiguous sounds_Surface contact_Grind_reference": 297,
        "298Source-ambiguous sounds_Surface contact_Roll_reference": 298,
        "299Source-ambiguous sounds_Surface contact_Rub_reference": 299,
        "300Source-ambiguous sounds_Surface contact_Scrape_reference": 300,
        "301Source-ambiguous sounds_Surface contact_Scratch_reference": 301,
    }
    config = TaskConfig(
        encoder=encoder,
        epochs=20,
        evalset_size=1867,
        formal_name="Vocal Imitation",
        k_fold_splits=list(range(0, 3)),
        label_processor=lambda x: class_label_maps[x[task]],
        name="vocalimitations",
        output_dim=len(class_label_maps),
        zenodo_id="14862060",
    )
    config.audio_tar_name_of_split = {
        fold: f"vocal_imitations_fold{fold:02}_0000000.tar" for fold in config.k_fold_splits
    }
    config.encoded_tar_name_of_split = {
        fold: f"vocalimiations-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
