package exceptions;
import model.disasters.*;
public abstract class DisasterException extends SimulationException{

	private Disaster disaster ;

	public DisasterException(Disaster disaster) {
		super();
		this.disaster=disaster;
	}
	public DisasterException(Disaster disaster,String message) {
		
		super(message);
		
	}
	public Disaster getDisaster() {
		return disaster;
	}
	
}
