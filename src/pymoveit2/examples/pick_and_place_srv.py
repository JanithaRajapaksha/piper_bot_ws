# ================================
# Motion logic
# ================================
def execute_sequence(self):
    try:
        self.get_logger().info("Waiting for TF...")

        transform = None
        start_time = time.time()

        # ----------------------------
        # Wait for TF
        # ----------------------------
        while rclpy.ok():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.target_frame,
                    rclpy.time.Time(),
                )
                break
            except Exception:
                if time.time() - start_time > 5.0:
                    self.get_logger().error("TF lookup timed out")
                    return
                time.sleep(0.1)

        if transform is None:
            self.get_logger().error("No TF received")
            return

        position = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z + 0.10,
        ]

        # ----------------------------
        # 1. Move to TF pose
        # ----------------------------
        try:
            self.get_logger().info(f"Moving to TF pose: {position}")
            self.moveit2.move_to_pose(
                position=position,
                quat_xyzw=self.quat_xyzw,
            )
            self.moveit2.wait_until_executed()
        except Exception as e:
            self.get_logger().error(f"Failed to move to TF pose: {e}")

        # ----------------------------
        # 2. Wait at pose
        # ----------------------------
        try:
            self.get_logger().info("Waiting at TF pose...")
            time.sleep(2.0)
        except Exception as e:
            self.get_logger().warn(f"Wait interrupted: {e}")

        # ----------------------------
        # 3. Predefined joint pose
        # ----------------------------
        try:
            self.get_logger().info("Moving to predefined joint pose")
            self.moveit2.move_to_configuration(self.predefined_joints)
            self.moveit2.wait_until_executed()
        except Exception as e:
            self.get_logger().error(f"Failed to move to predefined joints: {e}")

        # ----------------------------
        # 4. Zero position
        # ----------------------------
        try:
            self.get_logger().info("Moving to zero position")
            self.moveit2.move_to_configuration(self.zero_joints)
            self.moveit2.wait_until_executed()
        except Exception as e:
            self.get_logger().error(f"Failed to move to zero joints: {e}")

        self.get_logger().info("Sequence complete")

    finally:
        # Ensure the busy lock is released even if errors occur
        if self.busy_lock.locked():
            self.busy_lock.release()
