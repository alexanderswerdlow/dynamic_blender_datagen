from pathlib import Path
import os
import sys
import subprocess

DATA_DIR = Path.home() / "data" / "point_odyssey" / "data"
if DATA_DIR.exists() is False:
    DATA_DIR = Path("data")

DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))

def run_command(command):
    print(f"Running command: {command}")
    error_keywords = ("Error: Python: Traceback", "Error: Error")
    error_detected = False
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True) as proc:
        with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
            for line in proc.stdout:
                stdout.write(line)
                stdout.flush()
                if any(keyword in line.decode('utf-8') for keyword in error_keywords):
                    error_detected = True

        return_code = proc.wait()
        if return_code != 0:
            raise Exception(f"Command failed: {command}")
        
    if error_detected:
        raise Exception(f"Command failed: {command}")
        
urban_scenes = [
    'abandoned_bakery', 'abandoned_construction', 'abandoned_factory_canteen_01', 'abandoned_factory_canteen_02', 'abandoned_games_room_01', 'abandoned_games_room_02', 'abandoned_greenhouse', 'abandoned_hall_01', 'abandoned_parking', 'abandoned_slipway', 'abandoned_tank_farm_01', 'abandoned_tank_farm_02', 'abandoned_tank_farm_03', 'abandoned_tiled_room', 'abandoned_workshop', 'abandoned_workshop_02', 'acoustical_shell', 'adams_place_bridge', 'aerodynamics_workshop', 'aft_lounge', 'aircraft_workshop_01', 'altanka', 'amphitheatre_zanzibar_fort', 'anniversary_lounge', 'art_studio', 'artist_workshop', 'auto_service', 'autoshop_01', 'aviation_museum', 'balcony', 'ballroom', 'bank_vault', 'basement_boxing_ring', 'bathroom', 'beach_cloudy_bridge', 'beach_parking', 'belvedere', 'bethnal_green_entrance', 'between_bridges', 'binnenalster', 'birbeck_street_underpass', 'blaubeuren_church_square', 'blaubeuren_night', 'blender_institute', 'blinds', 'blocky_photo_studio', 'blue_lagoon', 'blue_lagoon_night', 'blue_photo_studio', 'boiler_room', 'brown_photostudio_01', 'brown_photostudio_02', 'brown_photostudio_03', 'brown_photostudio_04', 'brown_photostudio_05', 'brown_photostudio_06', 'brown_photostudio_07', 'buikslotermeerplein', 'burnt_warehouse', 'bush_restaurant', 'cabin', 'cambridge', 'canary_wharf', 'carpentry_shop_01', 'carpentry_shop_02', 'castel_st_angelo_roof', 'castle_zavelstein_cellar', 'cayley_interior', 'cedar_bridge', 'chapel_day', 'childrens_hospital', 'christmas_photo_studio_01', 'christmas_photo_studio_02', 'christmas_photo_studio_03', 'christmas_photo_studio_04', 'christmas_photo_studio_05', 'christmas_photo_studio_06', 'christmas_photo_studio_07', 'cinema_hall', 'cinema_lobby', 'circus_arena', 'cloudy_cliffside_road', 'cobblestone_street_night', 'colorful_studio', 'colosseum', 'combination_room', 'comfy_cafe', 'concrete_tunnel', 'concrete_tunnel_02', 'construction_yard', 'country_club', 'courtyard', 'courtyard_night', 'crosswalk', 'cyclorama_hard_light', 'dam_bridge', 'dancing_hall', 'de_balie', 'decor_shop', 'derelict_overpass', 'derelict_underpass', 'distribution_board', 'drachenfels_cellar', 'dresden_square', 'dresden_station_night', 'driving_school', 'dusseldorf_bridge', 'empty_warehouse_01', 'empty_workshop', 'en_suite', 'entrance_hall', 'factory_yard', 'fireplace', 'floral_tent', 'freight_station', 'furry_clouds', 'future_parking', 'garage', 'gear_store', 'georgentor', 'glass_passage', 'golden_bay', 'graffiti_shelter', 'gym_01', 'gym_entrance', 'hall_of_finfish', 'hall_of_mammals', 'hamburg_canal', 'hamburg_hbf', 'hangar_interior', 'hanger_exterior_cloudy', 'hansaplatz', 'hayloft', 'hospital_room', 'hospital_room_2', 'hotel_rooftop_balcony', 'hotel_room', 'indoor_pool', 'industrial_pipe_and_valve_01', 'industrial_pipe_and_valve_02', 'industrial_workshop_foundry', 'interior_construction', 'kart_club', 'kiara_interior', 'killesberg_park', 'konigsallee', 'konzerthaus', 'lake_pier', 'lapa', 'large_corridor', 'laufenurg_church', 'leadenhall_market', 'learner_park', 'lebombo', 'limehouse', 'little_paris_under_tower', 'lookout', 'lot_02', 'lythwood_lounge', 'lythwood_room', 'machine_shop_01', 'machine_shop_02', 'machine_shop_03', 'mall_parking_lot', 'marry_hall', 'medieval_cafe', 'metro_noord', 'metro_vijzelgracht', 'missile_launch_facility_01', 'modern_buildings', 'modern_buildings_2', 'modern_buildings_night', 'mosaic_tunnel', 'museum_of_ethnography', 'museum_of_history', 'museumplein', 'music_hall_01', 'music_hall_02', 'mutianyu', 'neon_photostudio', 'netball_court', 'neuer_zollhof', 'night_bridge', 'northcliff', 'old_apartments_walkway', 'old_bus_depot', 'old_depot', 'old_hall', 'old_outdoor_theater', 'old_room', 'orlando_stadium', 'outdoor_umbrellas', 'palermo_park', 'palermo_sidewalk', 'palermo_square', 'parking_garage', 'paul_lobe_haus', 'pedestrian_overpass', 'peppermint_powerplant', 'peppermint_powerplant_2', 'phone_shop', 'photo_studio_01', 'photo_studio_broadway_hall', 'photo_studio_loft_hall', 'photo_studio_london_hall', 'piazza_bologni', 'piazza_martin_lutero', 'piazza_san_marco', 'pillars', 'pine_attic', 'poly_haven_studio', 'pond_bridge_night', 'pool', 'portland_landing_pad', 'potsdamer_platz', 'preller_drive', 'pretville_cinema', 'pretville_street', 'provence_studio', 'pump_house', 'pump_station', 'pylons', 'quattro_canti', 'rathaus', 'reading_room', 'red_wall', 'reichstag_1', 'reinforced_concrete_01', 'reinforced_concrete_02', 'residential_garden', 'rhodes_memorial', 'roof_garden', 'roofless_ruins', 'rooftop_day', 'rooftop_night', 'rostock_arches', 'rostock_laage_airport', 'rotes_rathaus', 'round_platform', 'royal_esplanade', 'san_giuseppe_bridge', 'schadowplatz', 'school_hall', 'school_quad', 'sculpture_exhibition', 'sepulchral_chapel_basement', 'sepulchral_chapel_rotunda', 'shanghai_bund', 'shanghai_riverside', 'short_tunnel', 'signal_hill_dawn', 'signal_hill_sunrise', 'simons_town_harbour', 'skate_park', 'skidpan', 'skylit_garage', 'small_cathedral', 'small_cathedral_02', 'small_empty_house', 'small_empty_room_1', 'small_empty_room_2', 'small_empty_room_3', 'small_empty_room_4', 'small_hangar_01', 'small_hangar_02', 'small_workshop', 'snowy_cemetery', 'soliltude', 'solitude_interior', 'solitude_night', 'spree_bank', 'st_fagans_interior', 'st_peters_square_night', 'stadium_01', 'stone_alley', 'stone_alley_02', 'stone_alley_03', 'storeroom', 'street_lamp', 'studio_country_hall', 'studio_garden', 'studio_small_01', 'studio_small_02', 'studio_small_03', 'studio_small_04', 'studio_small_05', 'studio_small_06', 'studio_small_07', 'studio_small_08', 'studio_small_09', 'stuttgart_suburbs', 'suburban_parking_area', 'subway_entrance', 'summer_stage_01', 'summer_stage_02', 'sunset_jhbcentral', 'surgery', 'tears_of_steel_bridge', 'teatro_massimo', 'teufelsberg_ground_1', 'teufelsberg_ground_2', 'teufelsberg_inner', 'teufelsberg_lookout', 'teufelsberg_roof', 'thatch_chapel', 'the_sky_is_on_fire', 'theater_01', 'theater_02', 'tiber_1', 'tiber_2', 'tiber_island', 'trekker_monument', 'tv_studio', 'ulmer_muenster', 'under_bridge', 'unfinished_office', 'unfinished_office_night', 'urban_alley_01', 'urban_courtyard', 'urban_courtyard_02', 'urban_street_01', 'urban_street_02', 'urban_street_03', 'urban_street_04', 'vatican_road', 'venetian_crossroads', 'venice_dawn_1', 'venice_dawn_2', 'venice_sunrise', 'venice_sunset', 'veranda', 'vestibule', 'viale_giuseppe_garibaldi', 'vignaioli', 'vignaioli_night', 'vintage_measuring_lab', 'vulture_hide', 'whale_skeleton', 'wide_street_01', 'wide_street_02', 'winter_evening', 'wooden_lounge', 'wooden_motel', 'workshop', 'wrestling_gym', 'yaris_interior_garage', 'yoga_room', 'zavelstein', 'zhengyang_gate', 'zwinger_night'
]

validation_animals = [
    "tiger",
    "prisoner",
    "deer"
]

validation_blender_scenes = [
    "cab_e_ego2.blend",
    "scene_j716_3rd.blend",
]