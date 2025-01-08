import os
import pandas as pd
from utils_backend import *  

class ConfigBackend:
    def __init__(
        self,
        INPUT_FOLDER: str, 
        POSTJOB: bool = False,
        OPERATOR: str = "SIERRACOL",
        HOLE_DIAMETERS: list = [12.25, 8.5, 6.125],
        WELL_TYPE: str = "H",
        DLS_RANGE: list = ["any","any","any","0"],
        INC_RANGE: list = ["any","any","any","0","0"],
        NO_RT: bool = False,
        DTIME_RT = None,
        NULL_VALUE: float = -999.25,
        ROUND_NDIGITS: int = 3,
        REPLACE_DOT: str = "p"
    ):
        """
        Constructor de la clase. Define parámetros y variables globales.
        Carga la configuración desde el archivo JSON en el INPUT_FOLDER.
        
        Args:
            INPUT_FOLDER: Ruta base donde están los datos
        """
        # Cargar configuración del JSON predefinido
        config_file_path = os.path.join(INPUT_FOLDER, "inputs_for_rt.json")
        with open(config_file_path, 'r') as f:
            config_json = json.load(f)
        
        # Obtener valores del JSON
        self.CURRENT_WELL_NAME = config_json.get("CURRENT_WELL_NAME", "RT")
        self.WELLS_SELECT_NAME = config_json.get("WELLS_SELECT_NAME", [])
        current_bit_size = config_json.get("current_bit_size")
        
        # Si hay un bit_size en el JSON, ponerlo al inicio de HOLE_DIAMETERS
        if current_bit_size is not None:
            HOLE_DIAMETERS = [current_bit_size] + [
                d for d in HOLE_DIAMETERS if d != current_bit_size
            ]

        # ------------------------------------------------------------
        # 1. Parámetros principales
        # ------------------------------------------------------------
        self.POSTJOB = POSTJOB
        self.OPERATOR = OPERATOR
        
        # Ajustar la lista WELLS_SELECT_NAME para excluir el CURRENT_WELL_NAME
        self.WELLS_SELECT_NAME = [
            w for w in self.WELLS_SELECT_NAME if w != self.CURRENT_WELL_NAME
        ]

        self.HOLE_DIAMETERS = HOLE_DIAMETERS
        self.WELL_TYPE = WELL_TYPE
        self.DLS_RANGE = DLS_RANGE
        self.INC_RANGE = INC_RANGE

        # Parámetros de RT
        self.NO_RT = NO_RT
        self.DTIME_RT = DTIME_RT
        self.NULL_VALUE = NULL_VALUE

        # ------------------------------------------------------------
        # 2. Paths y rutas (usando el INPUT_FOLDER proporcionado)
        # ------------------------------------------------------------
        self.INPUT_FOLDER = INPUT_FOLDER
        self.SAVE_FOLDER = INPUT_FOLDER.replace("input_data", "output_data")
        self.SAVE_REAL_TIME_FOLDER = os.path.join(self.SAVE_FOLDER, "real_time_update")

        # ------------------------------------------------------------
        # 3. Parámetros de reporte y guardado
        # ------------------------------------------------------------
        if self.POSTJOB:
            # postjob
            self.report_folder = "postjob_report/"
            self.SAVE_XLSX = 1
            self.KEEP_RT_IN_HISTORIC = True
        else:
            # daily
            self.report_folder = "daily_report/"
            self.SAVE_XLSX = 0
            self.KEEP_RT_IN_HISTORIC = False

        self.SAVE_DAILY_REPORT_FOLDER = self.SAVE_FOLDER + self.report_folder
        self.DAILY_REPORT_PLOTS_PATH = self.SAVE_DAILY_REPORT_FOLDER.strip(".") + "plot/"

        # ------------------------------------------------------------
        # 4. Flags que dependen de POSTJOB y NO_RT
        # ------------------------------------------------------------
        self.READ_CURRENT_RIG_LABELS = False
        self.READ_BS_MM_CASING = False
        if self.POSTJOB or self.NO_RT:
            self.READ_CURRENT_RIG_LABELS = True
            self.READ_BS_MM_CASING = True

        # ------------------------------------------------------------
        # 5. Parámetros de formato
        # ------------------------------------------------------------
        self.ROUND_NDIGITS = ROUND_NDIGITS
        self.REPLACE_DOT = REPLACE_DOT
        self.STYLE = {"selector": "caption", "props": [("font-weight", "bold"), ("color", "k")]}

        # ------------------------------------------------------------
        # 6. Variables que se llenan al cargar los datos
        # ------------------------------------------------------------
        self.df_wells = None
        self.well_name_dict = None
        self.well_name_dict_swap = None
        self.CURRENT_WELL_ID = None
        self.WELLS_SELECT_ID = None
        self.RIG_NAME = None

        self.df_rig = None
        self.STAND_LENGTH = None

        # ------------------------------------------------------------
        # 7. Nombres de columnas (estándar)
        # ------------------------------------------------------------
        #Well info
        self.WELLID = 'well_id'
        self.WELL = 'well_name'
        self.WELLTYPE = 'well_type'
        self.WELLIDN = 'well_id_name'
        self.RIG = 'rig'
        self.FIELD = 'field'
        self.SPUD = 'spud_date'

        #Drilling parameters
        self.BS = 'hole_diameter'
        self.HD = 'measured_depth'
        self.BD = 'bit_depth'
        self.BH = 'block_height'
        self.WOB = 'wob'
        self.HL = 'hook_load'
        self.GPM = 'flow_rate'
        self.RPM = 'surface_rpm'
        self.MRPM = 'motor_rpm'
        self.BRPM = 'bit_rpm'
        self.TQ = 'torque'
        self.ROP = 'rop'
        self.SPP = 'spp'
        self.DIFFP = 'diff_pressure'
        self.PVT = 'pit_volume'
        self.APRESS = 'annular_pressure'
        self.HSI = 'hsi'

        #Logging curves
        self.GR = 'gamma'
        self.GRD = 'gamma_depth'

        #Formation tops
        self.FORM = 'formation'
        self.FORMTOP = 'formation_top_depth'
        self.FORMBOT = 'formation_bottom_depth'
        self.FORMTOPTIME = 'formation_top_time'
        self.FORMBOTTIME = 'formation_bottom_time'

        #Time
        self.DATETIME = 'datetime'
        self.DATETIMED = self.DATETIME + '_day'
        self.TIME = 'cumulative_time' #minutes
        self.TIMEDAY = 'cumulative_time_days' #days
        self.DAYN = 'day_number'
        self.MMDD = 'month_day'
        self.BITIMEFROM = 'bi_datetime_from'
        self.BITIMETO = 'bi_datetime_to'
        self.BISECFROM = 'bi_sections_datetime_from'
        self.BISECTO = 'bi_sections_datetime_to'
        self.CASINGTIMEFROM = 'casing_datetime_from'
        self.CASINGTIMETO = 'casing_datetime_to'

        #Consecutive labels
        self.LABEL = 'consecutive_labels'
        self.LABELct = self.LABEL + '_conn_trip'
        self.LABELcd = self.LABEL + '_conn_drill'
        self.LABELbtwn = self.LABEL + '_btwn'

        #Surveys
        self.DDI = 'ddi' #drilling difficulty index
        self.TVD = 'tvd'
        self.AZM = 'azm'
        self.INC = 'incl'
        self.NS = 'ns'
        self.EW = 'ew'
        self.VS = 'vs'
        self.DLS = 'dls'#(deg/100ft)
        self.WSEC = 'well_section'
        self.DDIR = 'ddi_range'
        self.UWDEP = 'unwrap_departure'
        self.TORT = 'tortuosity'
        self.ATORT = 'abs_tortuosity'
        self.DLSR = 'dls_range'
        self.INCR = 'inc_range'

        #Casing and motor
        self.CASING = 'casing'
        self.MM = 'mud_motor'

        #Calculate
        self.MSE = 'mse'
        self.WTWT = 'weight_to_weight'
        self.PIPESP = 'pipe_speed'
        self.PIPEMV = 'pipe_movement'
        self.TRIPSPEED = 'trip_speed'
        self.CONNDTIME = 'connection_time_drill'
        self.CONNTTIME = 'connection_time_trip'
        self.CONNTIME = 'connection_time'
        self.DD = 'depth_drilled'
        self.WCRC = 'wcr'

        #Enovate rig activities columns
        self.RSUPER = 'rig_super_state'
        self.RSUBACT = 'rig_sub_activity'

        #Auxiliary
        self.dBD = 'delta_' + self.BD
        self.dBH = 'delta_' + self.BH
        self.dTIME = 'delta_' + self.TIME
        self.DTIME_col = 'delta_' + self.TIME + '_days'
        self.DUR = 'duration'
        self.PLAN = 'plan'
        self.DEPTHTYPE = 'depth_analysis_type'
        self.DEPTHTVDTO = 'depth_analysis_tvd_to'
        self.COLOR = 'color'

        #drill tab
        self.STAND = 'stand_number'
        self.ROP_MSE = 'rop_mse'
        self.KPIC = 'kpi_color'

        #trip tab
        self.CIRCTIME = 'circulating_time'
        self.WASHTIME = 'washing_time'
        self.REAMTIME = 'reaming_time'

        #report
        self.GROUPWELL = 'group_wells'
        self.AVG = 'AVG'
        self.DIFF = 'DIFFERENCE'
        self.CLIENT = 'client'

        # ------------------------------------------------------------
        # 8. Conjuntos de columnas estándar
        # ------------------------------------------------------------
        self.rig_activity_order = [
            'ROTATE','SLIDE','REAM UP','REAM DOWN',
            'CNX (drill)','CNX (trip)','TRIP IN','TRIP OUT',
            'WASH IN','WASH OUT','PUMP ON/OFF','CIR (static)',
            'STATIC','OUT OF HOLE','OTHER','NULL'
        ]
        self.rig_activity_drill_order = [
            'ROTATE','SLIDE','REAM UP','REAM DOWN','CNX (drill)',
            'WASH IN','WASH OUT','PUMP ON/OFF','CIR (static)',
            'STATIC','OTHER','NULL'
        ]
        self.rig_activity_trip_order = [
            'REAM UP','REAM DOWN','CNX (trip)','TRIP IN','TRIP OUT',
            'WASH IN','WASH OUT','CIR (static)','STATIC','OTHER','NULL'
        ]

        self.time_based_drill_cols = [
            self.WELLID, self.DATETIME, self.TIME, self.DAYN,
            self.HD, self.TVD, self.INC, self.AZM, self.DLS, self.WSEC, 
            self.BD, self.BS, self.FORM, self.BH, self.ROP, self.WOB,
            self.HL, self.GPM, self.PVT, self.DIFFP, self.SPP, self.APRESS,
            self.TQ, self.RPM, self.MRPM, self.BRPM, self.MSE, self.MM,
            self.CASING, self.RSUPER, self.RSUBACT, self.LABEL
        ]

        self.time_based_drill_rt_raw_cols = [
            self.DATETIME, self.HD, self.BD, self.BH, self.ROP, self.WOB,
            self.HL, self.GPM, self.SPP, self.TQ, self.RPM, self.MRPM, self.BRPM
        ]

        self.survey_cols = [
            self.WELLID, self.HD, self.INC, self.AZM, self.TVD, self.DLS,
            self.NS, self.EW, self.TORT, self.ATORT, self.DDI, self.UWDEP,
            self.VS, self.WSEC, self.DDIR
        ]
        self.survey_raw_cols = [self.HD, self.INC, self.AZM]
        self.directional_plan_cols = [
            self.HD, self.TVD, self.INC, self.AZM, self.VS, self.NS, 
            self.EW, self.DLS, self.UWDEP, self.FORM, self.WSEC
        ]
        self.well_general_cols = [
            self.WELLID, self.WELL, self.WELLTYPE, self.CLIENT, self.RIG, 
            self.FIELD, self.SPUD, self.BITIMEFROM, self.BITIMETO, 'total_time', 
            'total_depth', 'location', 'latitude', 'longitude'
        ]
        self.rig_design_cols = [
            'rig_id','rig','client','stand_length','depth_onb_thr','depth_conn_thr',
            'depth_conn_start','bd_conn','depth_super_thr','depth_ooh_thr','depth_trip_thr',
            'hl_conn_drill_thr','hl_conn_drill1_thr','hl_conn_trip_thr','hl_conn_trip1_thr',
            'hl_null_thr','gpm_thr','rpm_thr','spp_thr','spp_stat_thr','wob_thr','rpm_rot_thr',
            'rpm_stat_thr','n_tseq_static','n_tseq_trip','n_tseq_circ','filter_dict',
            'filter_dict_1','filter_dict_5','filter_dict_10','filter_dict_15'
        ]
        self.formation_tops_plan_cols = [
            'well_name','formation','formation_top_depth','formation_top_tvd',
            'formation_top_time','formation_bottom_depth','formation_bottom_time'
        ]

        # ------------------------------------------------------------
        # 9. Colores y paletas
        # ------------------------------------------------------------
        self.color_overperf = '#9F8AD9'
        self.color_underperf = '#ce1140'
        self.color_neutral = '#2A2D40'
        self.color_historic = '#FCE255'
        self.color_RT = '#97FF8F'
        self.color_purple = '#9F8AD9'
        self.daily_report_color_RT = '#558A2E'

        self.kpi_colors = [
            self.color_underperf, self.color_underperf,
            self.color_neutral, self.color_overperf, self.color_overperf
        ]
        self.kpi_dict = {
            0: self.color_overperf,
            1: self.color_historic,
            2: self.color_underperf
        }
        self.perform_colors = [
            self.color_underperf, self.color_historic,
            self.color_overperf, self.color_neutral, self.color_RT
        ]
        self.mse_rop_colors = ['#583BD4','#9F8AD9','#F74B76','#CE1140']
        self.mse_rop_colors_map = dict(zip([0,1,2,3], self.mse_rop_colors))
        self.dd_range_color_dict = {
            '0-6': '#97FF8F','6-6.4': '#FFF500',
            '6.4-6.8': '#F5AB24','>6.8': '#CE1140'
        }
        self.well_diff_colors = ['#583BD4','#E3CB44','#841F38','#2A2D40']
        self.rig_activity_color_dict = {
            'ROTATE':'#FCE255','SLIDE':'#FDA300','REAM UP':'#C7B037','REAM DOWN':'#8C6C31',
            'CNX (drill)':'#F57524','CNX (trip)':'#E1C220','TRIP IN':'#9CCC66','TRIP OUT':'#558A2E',
            'WASH IN':'#80DDFF','WASH OUT':'#1EA5D5','PUMP ON/OFF':'#167A9D','CIR (static)':'#305580',
            'STATIC':'#9DAAB9','OUT OF HOLE':'#727272','OTHER':'#52587D','NULL':'#C8C8C8','TOTAL':None
        }
        self.act_backgr_dict = {
            key: ('background-color: ' + str(value))
            for key,value in self.rig_activity_color_dict.items()
        }

        # ------------------------------------------------------------
        # 10. Mapeo de super/sub actividades
        # ------------------------------------------------------------
        self.super_state = {0:'OTHER',1:'DRILL',7:'TRIP',5:'OUT OF HOLE'}
        self.super_state_swap = {v:k for k,v in self.super_state.items()}
        self.sub_activity_state = {
            0:'OTHER',1:'ROTATE',2:'SLIDE',3:'PUMP ON/OFF',
            5:'CNX (drill)',6:'CNX (trip)',7:'STATIC',8:'TRIP IN',
            9:'TRIP OUT',4:'NULL',11:'CIR (static)',12:'REAM UP',
            13:'REAM DOWN',14:'WASH IN',15:'WASH OUT',16:'OUT OF HOLE'
        }
        self.sub_activity_state_swap = {v:k for k,v in self.sub_activity_state.items()}
           
    # ------------------------------------------------------------
    # Métodos de carga de datos
    # ------------------------------------------------------------
    def load_well_general(self):
        """
        Carga el archivo well_general.csv, construye los diccionarios
        y actualiza las variables de CURRENT_WELL_ID, WELLS_SELECT_ID, RIG_NAME.
        """
        path_wells = os.path.join(self.INPUT_FOLDER, "database", "well_general.csv")
        self.df_wells = pd.read_csv(path_wells)

        # Diccionarios {well_id: well_name} y su inverso
        self.well_name_dict = {
            row[self.WELLID]: row[self.WELL]
            for _, row in self.df_wells.iterrows()
        }
        self.well_name_dict_swap = {v: k for k,v in self.well_name_dict.items()}

        # Asignar CURRENT_WELL_ID
        self.CURRENT_WELL_ID = self.well_name_dict_swap[self.CURRENT_WELL_NAME]
        
        # Asignar WELLS_SELECT_ID
        self.WELLS_SELECT_ID = [
            self.well_name_dict_swap[w] for w in self.WELLS_SELECT_NAME
        ]
        
        # Asignar RIG_NAME
        self.RIG_NAME = self.df_wells.loc[
            self.df_wells[self.WELLID] == self.CURRENT_WELL_ID, self.RIG
        ].values[0]

    def load_rig_design(self):
        """
        Carga el archivo rig_design.csv y asigna el STAND_LENGTH
        según el RIG_NAME detectado.
        """
        path_rig = os.path.join(self.INPUT_FOLDER, "database", "rig_design.csv")
        self.df_rig = pd.read_csv(path_rig)
        self.STAND_LENGTH = self.df_rig.loc[
            self.df_rig[self.RIG] == self.RIG_NAME, 'stand_length'
        ].values[0]

