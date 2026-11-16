"""Clasificación manual de zonas de WoW (2008) en tipos de actividad."""

CIUDADES = {"Orgrimmar", "Undercity", "Thunder Bluff", "Silvermoon City", "Shattrath City", "Dalaran",
            "Stormwind City", "Ironforge", "The Exodar", "Deeprun Tram"}

PVP = {"Arathi Basin", "Alterac Valley", "Warsong Gulch", "Eye of the Storm", "Strand of the Ancients",
       "Nagrand Arena", "Blade's Edge Arena", "Ruins of Lordaeron", "Dalaran競技場", "The Ring of Valor",
       "Wintergrasp", "Hall of Legends"}

RAIDS = {"Karazhan", "Serpentshrine Cavern", "Black Temple", "Hyjal", "Tempest Keep", "Gruul's Lair",
         "Magtheridon's Lair", "Sunwell Plateau", "Naxxramas", "The Obsidian Sanctum", "Vault of Archavon",
         "The Eye of Eternity", "Molten Core", "Blackwing Lair", "Onyxia's Lair", "Ahn'Qiraj",
         "Ruins of Ahn'Qiraj", "Zul'Gurub", "Zul'Aman"}

MAZMORRAS = {"The Mechanar", "Magisters' Terrace", "Hellfire Ramparts", "The Steamvault",
             "Coilfang: The Slave Pens", "The Botanica", "Auchindoun: Shadow Labyrinth", "The Underbog",
             "Scarlet Monastery", "Sethekk Halls", "The Blood Furnace", "The Shattered Halls", "The Arcatraz",
             "Mana-Tombs", "The Black Morass", "Zul'Farrak", "Old Hillsbrad Foothills", "Blackrock Depths",
             "Wailing Caverns", "Stratholme", "The Nexus", "The Temple of Atal'Hakkar", "Shadowfang Keep",
             "Razorfen Kraul", "Auchenai Crypts", "Scholomance", "The Culling of Stratholme", "The Violet Hold",
             "Ragefire Chasm", "Utgarde Pinnacle", "Halls of Lightning", "Utgarde Keep", "Razorfen Downs",
             "Maraudon", "Ahn'kahet: The Old Kingdom", "Dire Maul", "Gundrak", "Uldaman", "Drak'Tharon Keep",
             "Azjol-Nerub", "The Oculus", "Blackfathom Deeps", "Blackrock Spire", "Halls of Stone",
             "Gnomeregan", "Deadmines"}


def tipo_zona(z: str) -> str:
    if z in CIUDADES:
        return "ciudad"
    if z in PVP:
        return "pvp"
    if z in RAIDS:
        return "raid"
    if z in MAZMORRAS:
        return "mazmorra"
    return "mundo"
