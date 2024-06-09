package exceptions;
import model.units.*;
import simulation.*;
public class IncompatibleTargetException extends UnitException {

	
	public IncompatibleTargetException(Unit unit, Rescuable target) {
		super(unit,target);
	}
	public IncompatibleTargetException(Unit unit,Rescuable target,String message) {
		super(unit,target,message);
	}
	
	
}
