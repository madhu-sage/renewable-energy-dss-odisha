import geopandas as gpd
input_file  = r"C:\Users\KIIT0001\Desktop\Study Materials\ProjectX\renewable-energy-dss-odisha\phase6\odisha_energy_dss\data\blocks_complete.geojson"
output_file = r"C:\Users\KIIT0001\Desktop\Study Materials\ProjectX\renewable-energy-dss-odisha\phase6\odisha_energy_dss\data\blocks_complete.geojson"

blocks = gpd.read_file(input_file)

print("Current CRS:", blocks.crs)
print("Current bounds:", blocks.total_bounds)

# Convert if needed
if str(blocks.crs) != "EPSG:4326":
    blocks = blocks.to_crs("EPSG:4326")
    print("✅ Converted to EPSG:4326")
else:
    print("Already in EPSG:4326")

print("New bounds:", blocks.total_bounds)

# Save back
blocks.to_file(output_file, driver="GeoJSON")
print("✅ Done! File saved with correct projection.")