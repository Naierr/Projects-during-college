package model.events;

import simulation.Rescuable;
import simulation.Simulator;

public interface SOSListener {
public void receiveSOSCall(Rescuable r);

}
