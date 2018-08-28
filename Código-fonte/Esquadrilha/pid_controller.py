class PID:
    """
    Controle PID.
    kp: parametro proporcional
    ki: parametro integral
    kd: parametro diferencial
    min/max_value: valores maximo e minimo que o controle pode devolver na saida
    setpoint: valor desejado
    error: diferenca entre o valor atual e o valor desejado
    previous_error: valor do erro na iteracao anterior
    integrator: valor inicial do integrador
    integrator_max/min: valores maximo e minimo que o integrador pode assumir
    """

    def __init__(self, setpoint, P, I, D, integrator_max = 500, integrator_min = -500, max_value = 0.7, min_value = -0.7):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.min_value = min_value
        self.max_value = max_value
        self.setpoint = setpoint
        self.error = 0.0
        self.previous_error = 0.0
        self.integrator = 0.0
        self.integrator_max = integrator_max
        self.integrator_min = integrator_min

    def update(self, current_value):
        """
        Calcula a saida PID, tomando como entrada o valor atual.
        """

        self.error = current_value - self.setpoint
        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * (self.error - self.previous_error)
        self.previous_error = self.error
        self.integrator = self.integrator + self.error

        if self.integrator > self.integrator_max:
            self.integrator = self.integrator_max
        elif self.integrator < self.integrator_min:
            self.integrator = self.integrator_min

        self.I_value = self.integrator * self.Ki
        PID = self.P_value + self.I_value + self.D_value

        if PID > self.max_value:
            PID = self.max_value
        elif PID < self.min_value:
            PID = self.min_value

        return PID
