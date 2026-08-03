import netCDF4
import cdflib

def inspect_netcdf(nc_file_path):
    try:
        with netCDF4.Dataset(nc_file_path, mode='r') as nc:
            print(f"\nFile: {nc_file_path}")
            print("\nGlobal Attributes:")
            for attr in nc.ncattrs():
                print(f"  {attr}: {getattr(nc, attr)}")

            print("\nDimensions:")
            for dim in nc.dimensions.values():
                print(f"  {dim.name}: size={len(dim)}")

            #print(nc.variables["time"][:])

            print("\nVariables:")
            for var_name, var in nc.variables.items():
                print(f"  {var_name} ({var.dimensions}) -> shape: {var.shape}, dtype: {var.dtype}")
    except Exception as e:
        print(f"Error reading {nc_file_path}: {e}")

def inspect_cdf(file_path):
    file = cdflib.CDF(file_path)
    print(f"\nFile: {file_path}")
    print(file.cdf_info().zVariables)
    print(file.varget('Epoch')[0]-2600)
# Example usage
file_path = "data/sep-data/sis-data/1997/ac_h1_sis_19970829_v04.cdf"  # Replace with your path
inspect_cdf(file_path)


