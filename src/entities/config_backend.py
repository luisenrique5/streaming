import os
import json
import pandas as pd
from io import StringIO
from utils_backend import *  

class ConfigBackend:
    def __init__(
        self,
        base_key: str,      # Ej. 'input_data:client:project:stream:username:scenario'
        redis_client,       # Conexión a Redis
        POSTJOB: bool = False,
        OPERATOR: str = "SIERRACOL",
        HOLE_DIAMETERS: list = None,
        WELL_TYPE: str = "H",
        DLS_RANGE: list = None,
        INC_RANGE: list = None,
        NO_RT: bool = False,
        DTIME_RT = None,
        NULL_VALUE: float = -999.25,
        ROUND_NDIGITS: int = 3,
        REPLACE_DOT: str = "p"
    ):
        """
        Constructor de la clase. Define parámetros y variables globales.
        Carga la configuración desde Redis, en {base_key}:inputs_for_rt
        """
        self.base_key = base_key
        self.redis_client = redis_client

        # Leer la "configuración principal" desde Redis (análogo a inputs_for_rt.json)
        inputs_key = f"{self.base_key}:inputs_for_rt"
        inputs_data = redis_client.get(inputs_key)
        if not inputs_data:
            config_json = {}
            print(f"No se encontró la clave '{inputs_key}' en Redis. Se usarán defaults.")
        else:
            config_json = json.loads(inputs_data)

        # Del JSON, obtenemos CURRENT_WELL_NAME, WELLS_SELECT_NAME, current_bit_size
        self.CURRENT_WELL_NAME = config_json.get("CURRENT_WELL_NAME", "RT")
        self.WELLS_SELECT_NAME = config_json.get("WELLS_SELECT_NAME", [])
        current_bit_size = config_json.get("current_bit_size")

        # Si hay un current_bit_size, se coloca al inicio de HOLE_DIAMETERS
        if current_bit_size:
            if HOLE_DIAMETERS is None:
                HOLE_DIAMETERS = [current_bit_size]
            else:
                HOLE_DIAMETERS = [current_bit_size] + [d for d in HOLE_DIAMETERS if d != current_bit_size]
        if HOLE_DIAMETERS is None:
            HOLE_DIAMETERS = [12.25, 8.5, 6.125]
        if DLS_RANGE is None:
            DLS_RANGE = ["any", "any", "any", "0"]
        if INC_RANGE is None:
            INC_RANGE = ["any", "any", "any", "0", "0"]

        # 1. Parámetros principales
        self.POSTJOB = POSTJOB
        self.OPERATOR = OPERATOR
        self.WELLS_SELECT_NAME = [w for w in self.WELLS_SELECT_NAME if w != self.CURRENT_WELL_NAME]
        self.HOLE_DIAMETERS = HOLE_DIAMETERS
        self.WELL_TYPE = WELL_TYPE
        self.DLS_RANGE = DLS_RANGE
        self.INC_RANGE = INC_RANGE

        # Parámetros de RT
        self.NO_RT = NO_RT
        self.DTIME_RT = DTIME_RT
        self.NULL_VALUE = NULL_VALUE

        # 2. Paths y rutas
        # Extraemos client, project, stream, username, scenario del base_key (se asume formato correcto)
        splitted = base_key.split(":")
        if len(splitted) < 6 or splitted[0] != "input_data":
            raise ValueError(f"El base_key debe tener el formato 'input_data:client:project:stream:username:scenario': {base_key}")
        self.client = splitted[1]
        self.project = splitted[2]
        self.stream = splitted[3]
        self.username = splitted[4]
        self.scenario = splitted[5]

        # Creamos rutas locales válidas para guardado
        self.SAVE_FOLDER = os.path.join("output_data", self.client, self.project, self.stream, self.username, self.scenario)
        self.SAVE_REAL_TIME_FOLDER = os.path.join(self.SAVE_FOLDER, "real_time_update")
        # Se crean los directorios si no existen
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)
        os.makedirs(self.SAVE_REAL_TIME_FOLDER, exist_ok=True)

        # 3. Parámetros de reporte y guardado
        if self.POSTJOB:
            self.report_folder = "postjob_report"
            self.SAVE_XLSX = 1
            self.KEEP_RT_IN_HISTORIC = True
        else:
            self.report_folder = "daily_report"
            self.SAVE_XLSX = 0
            self.KEEP_RT_IN_HISTORIC = False

        self.SAVE_DAILY_REPORT_FOLDER = os.path.join(self.SAVE_FOLDER, self.report_folder)
        self.DAILY_REPORT_PLOTS_PATH = os.path.join(self.SAVE_DAILY_REPORT_FOLDER, "plot")
        os.makedirs(self.SAVE_DAILY_REPORT_FOLDER, exist_ok=True)
        os.makedirs(self.DAILY_REPORT_PLOTS_PATH, exist_ok=True)

        # 4. Flags dependientes de POSTJOB y NO_RT
        self.READ_CURRENT_RIG_LABELS = False
        self.READ_BS_MM_CASING = False
        if self.POSTJOB or self.NO_RT:
            self.READ_CURRENT_RIG_LABELS = True
            self.READ_BS_MM_CASING = True

        # 5. Parámetros de formato
        self.ROUND_NDIGITS = ROUND_NDIGITS
        self.REPLACE_DOT = REPLACE_DOT
        self.STYLE = {"selector": "caption", "props": [("font-weight", "bold"), ("color", "k")]}

        # 6. Variables que se llenan al cargar datos
        self.df_wells = None
        self.well_name_dict = None
        self.well_name_dict_swap = None
        self.CURRENT_WELL_ID = None
        self.WELLS_SELECT_ID = None
        self.RIG_NAME = None
        self.df_rig = None
        self.STAND_LENGTH = None

        # 7. Nombres de columnas (estándar)
        self.WELLID = 'well_id'
        self.WELL = 'well_name'
        self.WELLTYPE = 'well_type'
        self.WELLIDN = 'well_id_name'
        self.RIG = 'rig'
        self.FIELD = 'field'
        self.SPUD = 'spud_date'
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
        self.GR = 'gamma'
        self.GRD = 'gamma_depth'
        self.FORM = 'formation'
        self.FORMTOP = 'formation_top_depth'
        self.FORMBOT = 'formation_bottom_depth'
        self.FORMTOPTIME = 'formation_top_time'
        self.FORMBOTTIME = 'formation_bottom_time'
        self.DATETIME = 'datetime'
        self.DATETIMED = self.DATETIME + '_day'
        self.TIME = 'cumulative_time'
        self.TIMEDAY = 'cumulative_time_days'
        self.DAYN = 'day_number'
        self.MMDD = 'month_day'
        self.BITIMEFROM = 'bi_datetime_from'
        self.BITIMETO = 'bi_datetime_to'
        self.BISECFROM = 'bi_sections_datetime_from'
        self.BISECTO = 'bi_sections_datetime_to'
        self.CASINGTIMEFROM = 'casing_datetime_from'
        self.CASINGTIMETO = 'casing_datetime_to'
        self.LABEL = 'consecutive_labels'
        self.LABELct = self.LABEL + '_conn_trip'
        self.LABELcd = self.LABEL + '_conn_drill'
        self.LABELbtwn = self.LABEL + '_btwn'
        self.DDI = 'ddi'
        self.TVD = 'tvd'
        self.AZM = 'azm'
        self.INC = 'incl'
        self.NS = 'ns'
        self.EW = 'ew'
        self.VS = 'vs'
        self.DLS = 'dls'
        self.WSEC = 'well_section'
        self.DDIR = 'ddi_range'
        self.UWDEP = 'unwrap_departure'
        self.TORT = 'tortuosity'
        self.ATORT = 'abs_tortuosity'
        self.DLSR = 'dls_range'
        self.INCR = 'inc_range'
        self.CASING = 'casing'
        self.MM = 'mud_motor'
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
        self.RSUPER = 'rig_super_state'
        self.RSUBACT = 'rig_sub_activity'
        self.dBD = 'delta_' + self.BD
        self.dBH = 'delta_' + self.BH
        self.dTIME = 'delta_' + self.TIME
        self.DTIME_col = 'delta_' + self.TIME + '_days'
        self.DUR = 'duration'
        self.PLAN = 'plan'
        self.DEPTHTYPE = 'depth_analysis_type'
        self.DEPTHTVDTO = 'depth_analysis_tvd_to'
        self.COLOR = 'color'
        self.STAND = 'stand_number'
        self.ROP_MSE = 'rop_mse'
        self.KPIC = 'kpi_color'
        self.CIRCTIME = 'circulating_time'
        self.WASHTIME = 'washing_time'
        self.REAMTIME = 'reaming_time'
        self.GROUPWELL = 'group_wells'
        self.AVG = 'AVG'
        self.DIFF = 'DIFFERENCE'
        self.CLIENT = 'client'
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
        self.super_state = {0:'OTHER',1:'DRILL',7:'TRIP',5:'OUT OF HOLE'}
        self.super_state_swap = {v:k for k,v in self.super_state.items()}
        self.sub_activity_state = {
            0:'OTHER',1:'ROTATE',2:'SLIDE',3:'PUMP ON/OFF',
            5:'CNX (drill)',6:'CNX (trip)',7:'STATIC',8:'TRIP IN',
            9:'TRIP OUT',4:'NULL',11:'CIR (static)',12:'REAM UP',
            13:'REAM DOWN',14:'WASH IN',15:'WASH OUT',16:'OUT OF HOLE'
        }
        self.sub_activity_state_swap = {v:k for k,v in self.sub_activity_state.items()}

    def load_well_general(self):
        redis_key = f"{self.base_key}:database:well_general"
        data = self.redis_client.get(redis_key)
        if not data:
            print(f"No se encontró '{redis_key}' en Redis. No se asignarán well_name, etc.")
            return
        df_wells = pd.read_json(StringIO(data), orient="records")
        self.df_wells = df_wells
        self.well_name_dict = {row[self.WELLID]: row[self.WELL] for _, row in df_wells.iterrows()}
        self.well_name_dict_swap = {v: k for k, v in self.well_name_dict.items()}
        if self.CURRENT_WELL_NAME in self.well_name_dict_swap:
            self.CURRENT_WELL_ID = self.well_name_dict_swap[self.CURRENT_WELL_NAME]
        else:
            self.CURRENT_WELL_ID = None
        self.WELLS_SELECT_ID = [self.well_name_dict_swap[w] for w in self.WELLS_SELECT_NAME if w in self.well_name_dict_swap]
        if self.CURRENT_WELL_ID is not None:
            row = df_wells.loc[df_wells[self.WELLID] == self.CURRENT_WELL_ID]
            if not row.empty and self.RIG in row.columns:
                self.RIG_NAME = row[self.RIG].values[0]

    def load_rig_design(self):
        redis_key = f"{self.base_key}:database:rig_design"
        data = self.redis_client.get(redis_key)
        if not data:
            print(f"No se encontró '{redis_key}' en Redis. No se asigna STAND_LENGTH.")
            return
        df_rig = pd.read_json(StringIO(data), orient="records")
        self.df_rig = df_rig
        if self.RIG_NAME and not df_rig.empty and 'stand_length' in df_rig.columns:
            df_f = df_rig.loc[df_rig[self.RIG] == self.RIG_NAME]
            if not df_f.empty:
                self.STAND_LENGTH = df_f['stand_length'].values[0]
