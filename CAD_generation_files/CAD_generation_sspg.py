import sys
import os
import numpy as np

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import the module
import ActuatorAndGearbox as ActuatorAndGearbox

import pandas as pd
import json
# File path
file_path = os.path.join(parent_dir,'results','results_bilevel_U12','SSPG_BILEVEL_Lewis_U12.csv')
# Load CSV using pandas
data = pd.read_csv(file_path)

config_path = os.path.join(parent_dir, "config_files/config.json")
sspg_params_path = os.path.join(parent_dir, "config_files/sspg_params.json")

# Load the JSON file
with open(config_path, "r") as config_file:
    config_data = json.load(config_file)

with open(sspg_params_path, "r") as sspg_params_file:
    sspg_params = json.load(sspg_params_file)

sspg_design_params       = sspg_params["sspg_design_parameters"]
sspg_optimization_params = sspg_params["sspg_optimization_parameters"]
Gear_standard_parameters = config_data["Gear_standard_parameters"]


# Function to fetch gearbox parameters for a given gear ratio
def desired_gearbox(gear_ratio):
    row = data[data["gearRatio"] == gear_ratio]
    if not row.empty:
        return [sspg_design_params, Gear_standard_parameters,
            row.iloc[0]["Ns"], row.iloc[0]["Np"], row.iloc[0]["Nr"],
            row.iloc[0]["module"], row.iloc[0]["numPlanet"],
            row.iloc[0]["fwSunMM"], row.iloc[0]["fwPlanetMM"], row.iloc[0]["fwRingMM"]
        ]
    return None

# Define materials
steel = ActuatorAndGearbox.material(7800)
aluminum = ActuatorAndGearbox.material(2700)


motor_data          = config_data["Motors"]

#--------------------------------------------------------
# Motor Data
#--------------------------------------------------------
# T motor U8
MotorU8_maxTorque         = motor_data["MotorU8_framed"]["maxTorque"]             # Nm
MotorU8_power             = motor_data["MotorU8_framed"]["power"]                 # W 
MotorU8_maxMotorAngVelRPM = (MotorU8_power * 60) / (MotorU8_maxTorque * 2*np.pi) # RPM 
MotorU8_mass              = motor_data["MotorU8_framed"]["mass"]                  # kg 
MotorU8_dia               = motor_data["MotorU8_framed"]["dia"]                   # mm 
MotorU8_length            = motor_data["MotorU8_framed"]["length"]                # mm

# U10 Motor
MotorU10_maxTorque         = motor_data["MotorU10_framed"]["maxTorque"]              # Nm
MotorU10_power             = motor_data["MotorU10_framed"]["power"]                  # W
MotorU10_maxMotorAngVelRPM = (MotorU10_power * 60) / (MotorU10_maxTorque * 2*np.pi) # RPM
MotorU10_mass              = motor_data["MotorU10_framed"]["mass"]                   # Kg 
MotorU10_dia               = motor_data["MotorU10_framed"]["dia"]                    # mm
MotorU10_length            = motor_data["MotorU10_framed"]["length"]                 # mm

# MN8014 Motor
MotorMN8014_maxTorque         = motor_data["MotorMN8014_framed"]["maxTorque"]              # Nm
MotorMN8014_power             = motor_data["MotorMN8014_framed"]["power"]                  # W
MotorMN8014_maxMotorAngVelRPM = (MotorMN8014_power * 60) / (MotorMN8014_maxTorque * 2*np.pi) # RPM
MotorMN8014_mass              = motor_data["MotorMN8014_framed"]["mass"]                   # Kg 
MotorMN8014_dia               = motor_data["MotorMN8014_framed"]["dia"]                    # mm
MotorMN8014_length            = motor_data["MotorMN8014_framed"]["length"]                 # mm

# 8020 Motor
Motor8020_maxTorque         = motor_data["Motor8020_framed"]["maxTorque"]              # Nm
Motor8020_power             = motor_data["Motor8020_framed"]["power"]                  # W
Motor8020_maxMotorAngVelRPM = (Motor8020_power * 60) / (Motor8020_maxTorque * 2*np.pi) # RPM
Motor8020_mass              = motor_data["Motor8020_framed"]["mass"]                   # Kg 
Motor8020_dia               = motor_data["Motor8020_framed"]["dia"]                    # mm
Motor8020_length            = motor_data["Motor8020_framed"]["length"]                 # mm

# U12 Motor
MotorU12_maxTorque         = motor_data["MotorU12_framed"]["maxTorque"]             # Nm
MotorU12_power             = motor_data["MotorU12_framed"]["power"]                 # W 
MotorU12_maxMotorAngVelRPM = (MotorU12_power * 60) / (MotorU12_maxTorque * 2*np.pi) # RPM 
MotorU12_mass              = motor_data["MotorU12_framed"]["mass"]                  # kg 
MotorU12_dia               = motor_data["MotorU12_framed"]["dia"]                   # mm 
MotorU12_length            = motor_data["MotorU12_framed"]["length"]                # mm

# Motor-U8
MotorU8 = ActuatorAndGearbox.motor(maxMotorAngVelRPM = MotorU8_maxMotorAngVelRPM, 
                 maxMotorTorque    = MotorU8_maxTorque,
                 maxMotorPower     = MotorU8_power,
                 motorMass         = MotorU8_mass,
                 motorDia          = MotorU8_dia,
                 motorLength       = MotorU8_length,
                 motorName         = "U8")

# Motor-U10
MotorU10  = ActuatorAndGearbox.motor(maxMotorAngVelRPM   = MotorU10_maxMotorAngVelRPM, # RPM 
                  maxMotorTorque      = MotorU10_maxTorque,         # Nm 
                  maxMotorPower       = MotorU10_power,             # W 
                  motorMass           = MotorU10_mass,              # kg 
                  motorDia            = MotorU10_dia,               # mm 
                  motorLength         = MotorU10_length,     
                  motorName           = "U10") 

# Motor-MN8014
MotorMN8014 = ActuatorAndGearbox.motor(maxMotorAngVelRPM = MotorMN8014_maxMotorAngVelRPM, #RPM 
                    maxMotorTorque      = MotorMN8014_maxTorque      , # Nm 
                    maxMotorPower       = MotorMN8014_power          , # W 
                    motorMass           = MotorMN8014_mass           , # kg 
                    motorDia            = MotorMN8014_dia            , # mm 
                    motorLength         = MotorMN8014_length         ,     
                    motorName           = "MN8014") 

# VT8020 Motor
Motor8020 = ActuatorAndGearbox.motor(maxMotorAngVelRPM = Motor8020_maxMotorAngVelRPM , #RPM 
                  maxMotorTorque    = Motor8020_maxTorque         , # Nm 
                  maxMotorPower     = Motor8020_power             , # W 
                  motorMass         = Motor8020_mass              , # kg 
                  motorDia          = Motor8020_dia               , # mm 
                  motorLength       = Motor8020_length            ,
                  motorName         = "VT8020") 

# U12 Motor
MotorU12 = ActuatorAndGearbox.motor(maxMotorAngVelRPM = MotorU12_maxMotorAngVelRPM, 
                 maxMotorTorque    = MotorU12_maxTorque        , 
                 maxMotorPower     = MotorU12_power            ,
                 motorMass         = MotorU12_mass             , 
                 motorDia          = MotorU12_dia              , 
                 motorLength       = MotorU12_length           ,
                 motorName         = "U12")
# Input desired gear ratio
motor = MotorU12

desired_gr = 10
parameters = desired_gearbox(desired_gr)

if parameters is not None:
    # Initialize gearbox and actuator
    gearbox = ActuatorAndGearbox.singleStagePlanetaryGearbox(*parameters)
    actuator = ActuatorAndGearbox.singleStagePlanetaryActuator(sspg_design_params,motor, gearbox)

    # Print actuator parameters and generate equations
    actuator.getMassKG_new()
    actuator.genEquationFile()
else:
    print("Gear ratio not found.")
