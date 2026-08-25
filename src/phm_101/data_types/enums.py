from enum import Enum


class ImsTests(Enum):
    """IMS bearing tests.

    One parquet file per accelerometer, named ims_bearing_t1b3_x.
    """

    T1 = '1st_test'
    T2 = '2nd_test'
    T3 = '3rd_test'
