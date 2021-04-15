import numpy as np
import openmdao.api as om

from pycycle.thermo.thermo import Thermo

from pycycle.thermo.cea.species_data import janaf
from pycycle.constants import AIR_ELEMENTS
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough


class Coeff(om.ExplicitComponent):
    """
    Compute the coefficients of the fluid and coolant
    """
    def setup(self):
        self.add_input('W_fluid', units='lbm/s', desc='fluid mass flow')
        self.add_input('Cp_fluid', units='Btu/(lbm*degR)', desc='heat capacity at constant pressure of fluid')
        self.add_input('W_cool', units='lbm/s', desc='coolant mass flow')
        self.add_input('Cp_cool', units='Btu/(lbm*degR)', desc='heat capacity at constant pressure of coolant')

        self.add_output('C_max', units='Btu/(s*degR)', desc='minimum heat coefficient')
        self.add_output('C_min', units='Btu/(s*degR)', desc='maximum heat coefficient')
        self.add_output('C_r', desc='ratio heat coefficients')

        self.declare_partials('C_min', '*')
        self.declare_partials('C_max', '*')
        self.declare_partials('C_r', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        if inputs['W_fluid']*inputs['Cp_fluid'] > inputs['W_cool']*inputs['Cp_cool']:
            outputs['C_max'] = inputs['W_fluid']*inputs['Cp_fluid']
            outputs['C_min'] = inputs['W_cool']*inputs['Cp_cool']
            outputs['C_r'] = inputs['W_cool']*inputs['Cp_cool']/(inputs['W_fluid']*inputs['Cp_fluid'])
        else:
            outputs['C_min'] = inputs['W_fluid']*inputs['Cp_fluid']
            outputs['C_max'] = inputs['W_cool']*inputs['Cp_cool']
            outputs['C_r'] = inputs['W_fluid']*inputs['Cp_fluid']/(inputs['W_cool']*inputs['Cp_cool'])

    def compute_partials(self, inputs, J):
        if inputs['W_fluid']*inputs['Cp_fluid'] > inputs['W_cool']*inputs['Cp_cool']:
            J['C_max', 'W_fluid'] = inputs['Cp_fluid']
            J['C_max', 'Cp_fluid'] = inputs['W_fluid']
            J['C_min', 'W_cool'] = inputs['Cp_cool']
            J['C_min', 'Cp_cool'] = inputs['W_cool']
            J['C_r', 'W_fluid'] = -inputs['W_cool']*inputs['Cp_cool']/(inputs['W_fluid']**2*inputs['Cp_fluid'])
            J['C_r', 'Cp_fluid'] = -inputs['W_cool']*inputs['Cp_cool']/(inputs['W_fluid']*inputs['Cp_fluid']**2)
            J['C_r', 'W_cool'] = inputs['Cp_cool']/(inputs['W_fluid']*inputs['Cp_fluid'])
            J['C_r', 'Cp_cool'] = inputs['W_cool']/(inputs['W_fluid']*inputs['Cp_fluid'])
        else:
            J['C_min', 'W_fluid'] = inputs['Cp_fluid']
            J['C_min', 'Cp_fluid'] = inputs['W_fluid']
            J['C_max', 'W_cool'] = inputs['Cp_cool']
            J['C_max', 'Cp_cool'] = inputs['W_cool']
            J['C_r', 'W_fluid'] = inputs['Cp_fluid']/(inputs['W_cool']*inputs['Cp_cool'])
            J['C_r', 'Cp_fluid'] = inputs['W_fluid']/(inputs['W_cool']*inputs['Cp_cool'])
            J['C_r', 'W_cool'] = -inputs['W_fluid']*inputs['Cp_fluid']/(inputs['W_cool']**2*inputs['Cp_cool'])
            J['C_r', 'Cp_cool'] = -inputs['W_fluid']*inputs['Cp_fluid']/(inputs['W_cool']*inputs['Cp_cool']**2)


class NTUCalc(om.ExplicitComponent):
    """
    Compute the NTU
    """
    def setup(self):
        self.add_input('area', units='inch**2', desc='area of the heat exchanger')
        self.add_input('h_overall', units='Btu/(s*degR*inch**2)', desc='overall heat transfer coefficient')
        self.add_input('C_min', units='Btu/(s*degR)', desc='minimum heat coefficient')

        self.add_output('NTU', desc='NTU result')

        self.declare_partials('NTU', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        outputs['NTU'] = inputs['area']*inputs['h_overall']/inputs['C_min']

    def compute_partials(self, inputs, J):
        J['NTU', 'area'] = inputs['h_overall']/inputs['C_min']
        J['NTU', 'h_overall'] = inputs['area']/inputs['C_min']
        J['NTU', 'C_min'] = -inputs['area']*inputs['h_overall']/inputs['C_min']**2


class EffCalc(om.ExplicitComponent):
    """
    Calculate the effectiveness
    """
    def setup(self):
        self.add_input('W_fluid', units='lbm/s', desc='fluid mass flow')
        self.add_input('Cp_fluid', units='Btu/(lbm*degR)', desc='heat capacity at constant pressure of fluid')
        self.add_input('W_cool', units='lbm/s', desc='coolant mass flow')
        self.add_input('Cp_cool', units='Btu/(lbm*degR)', desc='heat capacity at constant pressure of coolant')

        self.add_input('C_r', desc='ratio heat coefficients')
        self.add_input('NTU', desc='NTU result')

        self.add_output('eff', desc='effectiveness result')

        self.declare_partials('eff', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        if inputs['W_fluid']*inputs['Cp_fluid'] > inputs['W_cool']*inputs['Cp_cool']:
            outputs['eff'] = 1-np.exp(-inputs['C_r']**(-1)*(1-np.exp(-inputs['C_r']*inputs['NTU'])))
        else:
            outputs['eff'] = (1/inputs['C_r'])*(1-np.exp(-inputs['C_r']*(1-np.exp(-inputs['NTU']))))

    def compute_partials(self, inputs, J):
        if inputs['W_fluid']*inputs['Cp_fluid'] > inputs['W_cool']*inputs['Cp_cool']:
            J['eff', 'C_r'] = -np.exp(-inputs['C_r']**(-1)*(1-np.exp(-inputs['C_r']*inputs['NTU'])))* \
                              (1-np.exp(-inputs['C_r']*inputs['NTU'])/inputs['C_r']**2-inputs['NTU']*np.exp(-inputs['C_r']*inputs['NTU'])/inputs['C_r'])
            J['eff', 'NTU'] = np.exp(-inputs['C_r']**(-1)*(1-np.exp(-inputs['C_r']*inputs['NTU']))-inputs['C_r']*inputs['NTU'])
        else:
            J['eff', 'C_r'] = -(1/inputs['C_r'])*(1-np.exp(-inputs['C_r']*(1-np.exp(-inputs['NTU']))))/inputs['C_r']**2 - \
                              (np.exp(-inputs['NTU'])-1)*np.exp(-inputs['C_r']*(1-np.exp(-inputs['NTU'])))/inputs['C_r']
            J['eff', 'NTU'] = np.exp(-inputs['C_r']*(1-np.exp(-inputs['NTU']))-inputs['NTU'])


class Qactual(om.ExplicitComponent):
    """
    Compute actual heat transfer rate
    """
    def setup(self):
        self.add_input('eff', desc='effectiveness result')
        self.add_input('C_min', units='Btu/(s*degR)', desc='minimum heat coefficient')
        self.add_input('T_fluid_in', units='degR', desc='temperature of incoming fluid')
        self.add_input('T_cool_in', units='degR', desc='temperature of incoming coolant')

        self.add_output('q_actual', units='Btu/lbm', desc='actual heat transfer per mass flow rate')

        self.declare_partials('q_actual', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        outputs['q_actual'] = inputs['eff']*inputs['C_min']*(inputs['T_fluid_in']-inputs['T_cool_in'])

    def compute_partials(self, inputs, J):
        J['q_actual', 'eff'] = inputs['C_min']*(inputs['T_fluid_in']-inputs['T_cool_in'])
        J['q_actual', 'C_min'] = inputs['eff']*(inputs['T_fluid_in']-inputs['T_cool_in'])
        J['q_actual', 'T_fluid_in'] = inputs['eff']*inputs['C_min']
        J['q_actual', 'T_cool_in'] = -inputs['eff']*inputs['C_min']


class TempChanges(om.ExplicitComponent):
    """
    Compute temperature changes of fluid and coolant
    """
    def setup(self):
        self.add_input('ht_in_fluid', units='Btu/lbm', desc='fluid incoming total enthalpy')
        self.add_input('ht_in_cool', units='Btu/lbm', desc='coolant incoming total enthalpy')
        self.add_input('q_actual', units='Btu/lbm', desc='heat ratio per mass flow rate')

        self.add_output('ht_out_fluid', units='Btu/lbm', desc='fluid outgoing total enthalpy')
        self.add_output('ht_out_cool', units='Btu/lbm', desc='coolant outgoing total enthalpy')

        self.declare_partials('ht_out_fluid', '*')
        self.declare_partials('ht_out_cool', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        outputs['ht_out_fluid'] = inputs['ht_in_fluid']-inputs['q_actual']
        outputs['ht_out_cool'] = inputs['ht_in_cool']+inputs['q_actual']

    def compute_partials(self, inputs, J):
        J['ht_out_fluid', 'ht_in_fluid'] = 1.
        J['ht_out_fluid', 'q_actual'] = -1.
        J['ht_out_cool', 'ht_in_cool'] = 1.
        J['ht_out_cool', 'q_actual'] = 1.


class PressureLoss(om.ExplicitComponent):
    """
    Calculates pressure loss across the heat exchanger.
    """

    def setup(self):
        self.add_input('dPqP', val=0.05,
                       desc='pressure differential as a fraction of incoming pressure')
        self.add_input('Pt_in_fluid', units='lbf/inch**2', desc='fluid inlet total pressure')
        self.add_input('Pt_in_cool', units='lbf/inch**2', desc='coolant inlet total pressure')

        self.add_output('Pt_out_fluid', units='lbf/inch**2', desc='fluid exit total pressure', lower=1e-3)
        self.add_output('Pt_out_cool', units='lbf/inch**2', desc='coolant exit total pressure', lower=1e-3)

        self.declare_partials('Pt_out_fluid', '*')
        self.declare_partials('Pt_out_cool', '*')
        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        outputs['Pt_out_fluid'] = inputs['Pt_in_fluid']*(1.0 - inputs['dPqP'])
        outputs['Pt_out_cool'] = inputs['Pt_in_cool']*(1.0 - inputs['dPqP'])

    def compute_partials(self, inputs, J):
        J['Pt_out_fluid', 'dPqP'] = -inputs['Pt_in_fluid']
        J['Pt_out_fluid', 'Pt_in_fluid'] = 1.0 - inputs['dPqP']
        J['Pt_out_cool', 'dPqP'] = -inputs['Pt_in_cool']
        J['Pt_out_cool', 'Pt_in_cool'] = 1.0 - inputs['dPqP']


class HeatExchanger(om.Group):
    """
    Calculates the outlet stations temperatures of the heat exchanger

    --------------
    Flow Stations
    --------------
    Fl_fluid_I
    Fl_fluid_O
    Fl_cool_I
    Fl_cool_O

    -------------
    Design
    -------------
        inputs
        --------
        Fl_fluid_I
        Fl_cool_I

        outputs
        --------
        Fl_fluid_O
        Fl_cool_O
    -------------
    Off-Design
    -------------
        inputs
        --------


        implicit states
        ---------------


    """

    def initialize(self):

        self.options.declare('thermo_data', default=janaf,
                             desc='thmodynamic data set', recordable=False)
        self.options.declare('Fl_I1_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the fluid')
        self.options.declare('Fl_I2_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the coolant')
        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design calculation.')
        self.options.declare('designed_stream', default=1, values=(1, 2),
                             desc='control for which stream has its area varied to match static pressure (1 means, you vary Fl_I1)')
        self.options.declare('internal_solver', default=True,
                             desc='If True, a newton solver is used inside the mixer to converge the impulse balance')

    def setup(self):

        thermo_data = self.options['thermo_data']
        fluid_elements = self.options['Fl_I1_elements']
        coolant_elements = self.options['Fl_I2_elements']
        statics = self.options['statics']
        design = self.options['design']

        in_flow = FlowIn(fl_name='Fl_I1')
        self.add_subsystem('in_flow_fluid', in_flow, promotes=['Fl_I1:tot:*', 'Fl_I1:stat:*'])  # 1 = fluid

        in_flow = FlowIn(fl_name='Fl_I2')
        self.add_subsystem('in_flow_cool', in_flow, promotes=['Fl_I2:tot:*', 'Fl_I2:stat:*'])  # 2 = coolant

        # Calculate the different coefficients
        prom_in = [('W_fluid', 'Fl_I1:stat:W'), ('Cp_fluid', 'Fl_I1:stat:Cp'), ('W_cool', 'Fl_I2:stat:W'), ('Cp_cool', 'Fl_I2:stat:Cp')]
        self.add_subsystem('coeff', Coeff(), promotes_inputs=prom_in)

        # Calculate the NTU
        prom_in = ['area', 'h_overall', 'C_min']
        self.add_subsystem('ntu', NTUCalc(), promotes_inputs=prom_in)

        # Calculate the efficiency
        prom_in = [('W_fluid', 'Fl_I1:stat:W'), ('Cp_fluid', 'Fl_I1:stat:Cp'), ('W_cool', 'Fl_I2:stat:W'), ('Cp_cool', 'Fl_I2:stat:Cp'), 'C_r', 'NTU']
        self.add_subsystem('eff_calc', EffCalc(), promotes_inputs=prom_in)

        # Calculate actual heat transfer rate
        prom_in = ['eff', 'C_min', ('T_fluid_in', 'Fl_I1:stat:T'), ('T_cool_in', 'Fl_I2:stat:T')]
        self.add_subsystem('q_calc', Qactual(), promotes_inputs=prom_in)

        # Calculate fluid and coolant temperature changes
        prom_in = [('ht_in_fluid', 'Fl_I1:tot:h'), ('ht_in_cool', 'Fl_I2:tot:h'), 'q_actual']
        self.add_subsystem('temp_changes', TempChanges(), promotes_inputs=prom_in)

        # Calculate fluid and coolant pressure changes
        prom_in = [('Pt_in_fluid', 'Fl_I1:tot:P'), ('Pt_in_cool', 'Fl_I2:tot:P'), 'dPqP']
        self.add_subsystem('p_loss', PressureLoss(), promotes_inputs=prom_in)

        # Connect all calculations
        self.connect('coeff.C_min', 'C_min')
        self.connect('coeff.C_r', 'C_r')
        self.connect('ntu.NTU', 'NTU')
        self.connect('eff_calc.eff', 'eff')
        self.connect('q_calc.q_actual', 'q_actual')

        # Total Calc
        real_flow_fluid = Thermo(mode='total_hP', fl_name='Fl_O1:tot',
                           method='CEA',
                           thermo_kwargs={'elements': fluid_elements,
                                          'spec': thermo_data})
        prom_in = [('composition', 'Fl_I1:tot:composition')]
        self.add_subsystem('real_flow_fluid', real_flow_fluid, promotes_inputs=prom_in,
                           promotes_outputs=['Fl_O1:*'])
        self.connect('temp_changes.ht_out_fluid', 'real_flow_fluid.h')
        self.connect('p_loss.Pt_out_fluid', 'real_flow_fluid.P')

        real_flow_cool = Thermo(mode='total_hP', fl_name='Fl_O2:tot',
                                 method='CEA',
                                 thermo_kwargs={'elements':coolant_elements,
                                                'spec': thermo_data})
        prom_in = [('composition', 'Fl_I2:tot:composition')]
        self.add_subsystem('real_flow_cool', real_flow_cool, promotes_inputs=prom_in,
                           promotes_outputs=['Fl_O2:*'])
        self.connect('temp_changes.ht_out_cool', 'real_flow_cool.h')
        self.connect('p_loss.Pt_out_cool', 'real_flow_cool.P')

        if statics:
            if design:
                #   Calculate static properties
                out_stat_fluid = Thermo(mode='static_A', fl_name='Fl_O1:stat',
                                  method='CEA',
                                  thermo_kwargs={'elements': fluid_elements,
                                                 'spec': thermo_data})
                prom_in = [('composition', 'Fl_I1:tot:composition'),
                           ('W', 'Fl_I1:stat:W'), ('area', 'Fl_I1:stat:area')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out_stat_fluid', out_stat_fluid, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O1:tot:S', 'out_stat_fluid.S')
                self.connect('Fl_O1:tot:h', 'out_stat_fluid.ht')
                self.connect('Fl_O1:tot:P', 'out_stat_fluid.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out_stat_fluid.guess:gamt')

                out_stat_cool = Thermo(mode='static_A', fl_name='Fl_O2:stat',
                                        method='CEA',
                                        thermo_kwargs={'elements': coolant_elements,
                                                       'spec': thermo_data})
                prom_in = [('composition', 'Fl_I2:tot:composition'),
                           ('W', 'Fl_I2:stat:W'), ('area', 'Fl_I2:stat:area')]
                prom_out = ['Fl_O2:stat:*']
                self.add_subsystem('out_stat_cool', out_stat_cool, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O2:tot:S', 'out_stat_cool.S')
                self.connect('Fl_O2:tot:h', 'out_stat_cool.ht')
                self.connect('Fl_O2:tot:P', 'out_stat_cool.guess:Pt')
                self.connect('Fl_O2:tot:gamma', 'out_stat_cool.guess:gamt')
            else:
                # Calculate static properties
                out_stat_fluid = Thermo(mode='static_A', fl_name='Fl_O1:stat',
                                  method='CEA',
                                  thermo_kwargs={'elements': fluid_elements,
                                                 'spec': thermo_data})
                prom_in = [('composition', 'Fl_I1:tot:composition'),
                           ('W', 'Fl_I1:stat:W'), ('area', 'Fl_I1:stat:area')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out_stat_fluid', out_stat_fluid, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O1:tot:S', 'out_stat_fluid.S')
                self.connect('Fl_O1:tot:h', 'out_stat_fluid.ht')
                self.connect('Fl_O1:tot:P', 'out_stat_fluid.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out_stat_fluid.guess:gamt')

                out_stat_cool = Thermo(mode='static_A', fl_name='Fl_O2:stat',
                                        method='CEA',
                                        thermo_kwargs={'elements': coolant_elements,
                                                       'spec': thermo_data})
                prom_in = [('composition', 'Fl_I2:tot:composition'),
                           ('W', 'Fl_I2:stat:W'), ('area', 'Fl_I2:stat:area')]
                prom_out = ['Fl_O2:stat:*']
                self.add_subsystem('out_stat_cool', out_stat_cool, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O2:tot:S', 'out_stat_cool.S')
                self.connect('Fl_O2:tot:h', 'out_stat_cool.ht')
                self.connect('Fl_O2:tot:P', 'out_stat_cool.guess:Pt')
                self.connect('Fl_O2:tot:gamma', 'out_stat_cool.guess:gamt')
        else:
            self.add_subsystem('W_passthru_fluid', PassThrough('Fl_I1:stat:W', 'Fl_O1:stat:W', 1.0, units="lbm/s"), promotes=['*'])
            self.add_subsystem('W_passthru_cool', PassThrough('Fl_I2:stat:W', 'Fl_O2:stat:W', 1.0, units="lbm/s"), promotes=['*'])