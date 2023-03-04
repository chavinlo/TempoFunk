LOBBY!!!!

02/03/2023 - 3 DE MARZO DE 2023

LR:
El lr mas efectivo hasta ahora es 1e-4. Solo lo probe con el Adam optimizer pero con el LION debe ser el mismo dividido por 3 (1e-4/3)

DATA:
Primero intentamos con un dataset pequeño de un canal de tiktok, con un label estatico de "dance, coreography". el performance fue decente pero no daba tanto potencial, ademas que tenia un limite bien pequeño de aprendizaje, se overfiteaba rapido

Ahora estamos usando una porcion de WebVid, un dataset grande de videos, solo cojimos los videos que contienen las palabras "dance", "coreography", y "performance"
Este nuevo dataset esta funcionando de maravilla, el modelo no logra overfitearse como se puede ver en el analisis de loss_lr. aunque el loss normal esta bajando muy lentamente, esta funcionando, y segun mi analisis del loss_lr, aun queda mucho para que se overfitee.

CONSEJOS:
lxj616 (alias el chino) dice que podriamos utilizar, en lugar de atencion normal, atencion local. Segun el funciona mejor y va mas rapido que la atencion normal. Aunasi, menciono que tiene la posible limitacion de, a la hora de inferencia o generacion, el video resultante va a repetir la misma accion una y otra vez.

Adicionalmente, pienso yo si seria posible usar XFormers ya que no veo que este siendo utilizado, creo que esta habilita-ble para todos los modulos excepto los nuevos, estos siendo los de la atencion en el pseudo3dunet

Tambien talvez utilizar el modulo ese de WANDB? ese el que dice que puede automaticamente encontrar los hyper-parametros mas eficientes.

tambien hay que preguntale al graham si nos puede dar mas gpus xD hay que ir mas rapido 

CONCLUSION:
El modelo esta empezando a demostrar que tiene potencial
Deje una sesion de entrenamiento corriendo con el nuevo LION, a ver como va.