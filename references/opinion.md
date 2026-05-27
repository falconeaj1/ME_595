## Opinion: When Should ML or AI Be Used in Feedback Control?

**Submitted by :** Patrick Smith

---

Machine learning and AI can improve a control system in most cases, but whether the improvement justifies the cost and complexity should be carefully considered. A thermostat using a simple on/off rule does not need a neural network. A self-driving car navigating an unstructured environment at human scale may have no viable alternative to AI. A structured framework helps determine where a given application falls.

**Define requirements before choosing methodology.** The system's requirements, objectives, and constraints should be established before evaluating any methodology. What performance is needed, under what conditions, at what cost, and with what reliability? Are there constraints on processors, weight, or power? Defining these upfront avoids selecting a methodology because it is interesting or fashionable, and provides an honest basis for comparing options.

**Is the solution cost effective?** If an ML-enhanced thermostat saves $40 per year but requires a more capable processor, cloud connectivity, and ongoing model maintenance, it may never pay for itself. The efficiency gain must justify the training cost, hardware cost, and operational overhead. When the performance gap between AI and classical methods is small, simpler is almost always better.

**Is the data available?** ML-based control requires data for training and for inference, and both availability and quality matter. What sensors are available, at what rate, and with what noise? A drone with GPS, IMU, and onboard cameras has rich, high-rate state information well suited to learning-based control. An industrial process with slow, expensive sensors may not. If the data needed cannot be obtained reliably within budget, AI-based control is not viable regardless of its theoretical advantages.

**Can inference fit the deployment constraints?** Training typically happens offline on a GPU cluster. But inference must run wherever the controller lives: onboard a microcontroller, inside a real-time PLC, or on a server with a network in the loop. A controller that requires a cloud connection introduces latency and a single point of failure that may be unacceptable. Compute, power, memory, and weight constraints are design constraints that should eliminate options early.

**Can the approach meet safety and certification requirements?** Certification requires that a controller's behavior be explainable and its failure modes bounded. Neural network policies are difficult to certify because their decisions cannot be inspected and their stability cannot be formally proven, properties that classical controllers such as PID, LQR, and MPC satisfy by design. If an ML approach cannot be made interpretable enough, it is worth asking whether requirements can be responsibly scoped to still meet the project intent, or whether a classical approach is simply the right answer.

The right question is not "can we use AI here?" but "should we?" When the improvement is large, the data is available, deployment constraints are satisfied, classical alternatives fall short, and certification requirements can be met, AI-based control may be the only viable path. When those conditions are not met, simpler is usually better. AI is not a substitute for sound engineering judgment.